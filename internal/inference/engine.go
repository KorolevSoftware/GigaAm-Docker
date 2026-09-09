package inference

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/diagnostics"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
	ort "github.com/yalue/onnxruntime_go"
)

type Engine struct {
	c                            config.Config
	features                     *Features
	vocab                        []string
	blank                        int
	encoder, decoder, joint, vad *graph
}

func New(ctx context.Context, c config.Config, asrDir, vadDir string) (*Engine, error) {
	// Disable process telemetry before initialization, including POSIX builds.
	if err := os.Setenv("ORT_DISABLE_TELEMETRY", "1"); err != nil {
		return nil, diagnostics.Wrap("telemetry_setup_failed", err)
	}
	ort.SetSharedLibraryPath(c.ORTLibrary)
	if e := ort.InitializeEnvironment(); e != nil {
		return nil, diagnostics.Wrap("runtime_load_failed", e)
	}
	e := &Engine{c: c, features: NewFeatures()}
	success := false
	defer func() {
		if !success {
			e.Close()
		}
	}()
	if err := ort.DisableTelemetry(); err != nil {
		return nil, diagnostics.Wrap("telemetry_setup_failed", err)
	}
	logging.Log.Info().Str("version", ort.GetVersion()).Str("telemetry", "disabled").Msg("ONNX Runtime initialized")
	prefix := "v3_e2e_rnnt"
	if strings.HasSuffix(c.Model, "ctc") {
		prefix = "v3_e2e_ctc"
	}
	var err error
	e.vocab, e.blank, err = loadVocab(filepath.Join(asrDir, prefix+"_vocab.txt"))
	if err != nil {
		return nil, diagnostics.Wrap("vocabulary_load_failed", err)
	}
	if prefix == "v3_e2e_ctc" {
		e.encoder, err = openGraph(filepath.Join(asrDir, prefix+".onnx"), []string{"features", "feature_lengths"}, []string{"log_probs"}, c.Threads)
	} else {
		e.encoder, err = openGraph(filepath.Join(asrDir, prefix+"_encoder.onnx"), []string{"audio_signal", "length"}, []string{"encoded", "encoded_len"}, c.Threads)
		if err == nil {
			e.decoder, err = openGraph(filepath.Join(asrDir, prefix+"_decoder.onnx"), []string{"x", "h.1", "c.1"}, []string{"dec", "h", "c"}, c.Threads)
		}
		if err == nil {
			e.joint, err = openGraph(filepath.Join(asrDir, prefix+"_joint.onnx"), []string{"enc", "dec"}, []string{"joint"}, c.Threads)
		}
	}
	if err != nil {
		return nil, diagnostics.Wrap("asr_graph_load_failed", err)
	}
	e.vad, err = openGraph(filepath.Join(vadDir, "silero_vad.onnx"), []string{"input", "state", "sr"}, []string{"output", "stateN"}, 1)
	if err != nil {
		return nil, diagnostics.Wrap("vad_load_failed", err)
	}
	if _, err = e.recognize(ctx, make([]float32, 16000)); err != nil {
		return nil, diagnostics.Wrap("asr_warmup_failed", err)
	}
	v := vadState{state: make([]float32, 2*128)}
	if _, err = e.speech(ctx, make([]float32, 512), &v); err != nil {
		return nil, diagnostics.Wrap("vad_warmup_failed", err)
	}
	success = true
	return e, nil
}
func (e *Engine) Close() {
	e.encoder.close()
	e.decoder.close()
	e.joint.close()
	e.vad.close()
	_ = ort.DestroyEnvironment()
}
func loadVocab(path string) ([]string, int, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, 0, e
	}
	defer f.Close()
	var vocab []string
	blank := -1
	s := bufio.NewScanner(f)
	for s.Scan() {
		line := s.Text()
		pos := strings.LastIndexByte(line, ' ')
		if pos < 0 {
			return nil, 0, fmt.Errorf("invalid vocabulary")
		}
		id, err := strconv.Atoi(line[pos+1:])
		if err != nil || id != len(vocab) {
			return nil, 0, fmt.Errorf("invalid vocabulary index")
		}
		piece := line[:pos]
		if piece == "<blk>" {
			blank = id
		}
		vocab = append(vocab, piece)
	}
	if err := s.Err(); err != nil {
		return nil, 0, err
	}
	if blank != len(vocab)-1 || blank < 1 {
		return nil, 0, fmt.Errorf("invalid blank token")
	}
	return vocab, blank, nil
}
func (e *Engine) decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id < 0 || id >= e.blank {
			continue
		}
		p := e.vocab[id]
		switch p {
		case "<unk>":
			b.WriteString(" ⁇ ")
		case "<s>", "</s>":
		default:
			b.WriteString(strings.ReplaceAll(p, "▁", " "))
		}
	}
	return strings.TrimPrefix(b.String(), " ")
}
func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}
func (e *Engine) recognizeTokens(ctx context.Context, x []float32) ([]emission, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	m := metrics.FromContext(ctx)
	start := time.Now()
	features, n := e.features.Compute(x)
	m.AddDuration(metrics.Frontend, time.Since(start))
	start = time.Now()
	out, err := e.encoder.run(ctx, ft([]int64{1, 64, int64(n)}, features), it([]int64{1}, []int64{int64(n)}))
	m.AddDuration(metrics.Encoder, time.Since(start))
	if err != nil {
		return nil, err
	}
	start = time.Now()
	defer func() { m.AddDuration(metrics.Decoder, time.Since(start)) }()
	if e.decoder == nil {
		shape := out[0].shape
		if len(shape) != 3 || shape[0] != 1 || shape[2] != int64(len(e.vocab)) {
			return nil, fmt.Errorf("invalid CTC output shape")
		}
		frames := int(shape[1])
		if limit := (n-1)/4 + 1; frames > limit {
			frames = limit
		}
		var ids []emission
		prev := -1
		for t := 0; t < frames; t++ {
			id := argmax(out[0].f[t*len(e.vocab) : (t+1)*len(e.vocab)])
			if id != e.blank && id != prev {
				ids = append(ids, emission{id: id, sample: int64(t * 640)})
			}
			prev = id
		}
		return ids, nil
	}
	shape := out[0].shape
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 768 || len(out[1].i) != 1 {
		return nil, fmt.Errorf("invalid RNNT output shape")
	}
	frames := int(shape[2])
	valid := int(out[1].i[0])
	if valid < 0 || valid > frames {
		return nil, fmt.Errorf("invalid encoded length")
	}
	h, c := make([]float32, 320), make([]float32, 320)
	last := int64(e.blank)
	var ids []emission
	var prediction []tensor
	for t := 0; t < valid; t++ {
		f := make([]float32, 768)
		for k := range f {
			f[k] = out[0].f[k*frames+t]
		}
		for emitted := 0; emitted < 3; emitted++ {
			if prediction == nil {
				prediction, err = e.decoder.run(ctx, it([]int64{1, 1}, []int64{last}), ft([]int64{1, 1, 320}, h), ft([]int64{1, 1, 320}, c))
				if err != nil {
					return nil, err
				}
				if len(prediction[0].f) != 320 || len(prediction[1].f) != 320 || len(prediction[2].f) != 320 {
					return nil, fmt.Errorf("invalid prediction shape")
				}
			}
			j, err := e.joint.run(ctx, ft([]int64{1, 768, 1}, f), ft([]int64{1, 320, 1}, prediction[0].f))
			if err != nil {
				return nil, err
			}
			if len(j[0].f) != len(e.vocab) {
				return nil, fmt.Errorf("invalid joint shape")
			}
			id := argmax(j[0].f)
			if id == e.blank {
				break
			}
			ids = append(ids, emission{id: id, sample: int64(t * 640)})
			last = int64(id)
			h, c = prediction[1].f, prediction[2].f
			prediction = nil
		}
	}
	return ids, nil
}

type emission struct {
	id     int
	sample int64
}

func (e *Engine) recognize(ctx context.Context, x []float32) (string, error) {
	tokens, err := e.recognizeTokens(ctx, x)
	if err != nil {
		return "", err
	}
	return e.decodeTimed(tokens), nil
}
func (e *Engine) decodeTimed(tokens []emission) string {
	ids := make([]int, len(tokens))
	for i, t := range tokens {
		ids[i] = t.id
	}
	return e.decode(ids)
}

// Approximate word start positions come from emitted encoder frames. They
// bound overlap matching; identical phrases elsewhere must never be removed.
func (e *Engine) wordPositions(tokens []emission) []int64 {
	var positions []int64
	inWord := false
	for _, t := range tokens {
		piece := strings.ReplaceAll(e.vocab[t.id], "▁", " ")
		if piece == "<unk>" {
			piece = " ⁇ "
		}
		if piece == "<s>" || piece == "</s>" {
			continue
		}
		for _, r := range piece {
			if unicode.IsSpace(r) {
				inWord = false
			} else if !inWord {
				positions = append(positions, t.sample)
				inWord = true
			}
		}
	}
	return positions
}

type vadState struct {
	state []float32
	tail  [64]float32
}

func (e *Engine) speech(ctx context.Context, x []float32, v *vadState) (float32, error) {
	input := make([]float32, 576)
	copy(input, v.tail[:])
	copy(input[64:], x)
	copy(v.tail[:], input[512:])
	out, err := e.vad.run(ctx, ft([]int64{1, 576}, input), ft([]int64{2, 1, 128}, v.state), it([]int64{}, []int64{16000}))
	if err != nil {
		return 0, err
	}
	if len(out[0].f) != 1 || len(out[1].f) != 256 {
		return 0, fmt.Errorf("invalid VAD output")
	}
	v.state = out[1].f
	return out[0].f[0], nil
}
func (e *Engine) Transcribe(ctx context.Context, w *audio.WAV) (string, error) {
	m := metrics.FromContext(ctx)
	chunks, asrSamples := 0, int64(0)
	defer func() {
		m.RecordASR(chunks, float64(asrSamples)/audio.Rate)
	}()
	v := vadState{state: make([]float32, 256)}
	chunk := int64(e.c.Chunk.Seconds() * audio.Rate)
	overlap := int64(e.c.Overlap.Seconds() * audio.Rate)
	pad := int64(e.c.Padding.Seconds() * audio.Rate)
	silence := int64(e.c.MinSilence.Seconds() * audio.Rate)
	start, lastVoice, silenceAt := int64(-1), int64(0), int64(-1)
	previousEnd := int64(-1)
	var result strings.Builder
	lastText := ""
	var previousWords []int64
	previousStart := int64(0)
	emit := func(a, b int64) error {
		if b <= a {
			return nil
		}
		samples, err := w.ReadSamples(a, int(b-a))
		if err != nil {
			return err
		}
		chunks++
		asrSamples += b - a
		tokens, err := e.recognizeTokens(ctx, samples)
		if err != nil {
			return err
		}
		text := strings.TrimSpace(e.decodeTimed(tokens))
		words := e.wordPositions(tokens)
		if text != "" {
			addition := text
			if a < previousEnd {
				suffix, prefix := 0, 0
				for _, p := range previousWords {
					if previousStart+p >= a-3200 {
						suffix++
					}
				}
				if suffix < len(previousWords) {
					suffix++
				} // word crossing the left edge
				for _, p := range words {
					if a+p <= previousEnd+1600 {
						prefix++
					}
				}
				addition = overlapSuffix(lastText, text, min(suffix, prefix))
			}
			if addition != "" {
				if result.Len() > 0 {
					result.WriteByte(' ')
				}
				result.WriteString(addition)
			}
		}
		lastText = text
		previousWords = words
		previousStart = a
		previousEnd = b
		return nil
	}
	// Only a 512-sample VAD frame and one <=30-second ASR window are resident.
	for pos := int64(0); pos < w.Samples; pos += 512 {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		x, err := w.ReadSamples(pos, 512)
		if err != nil {
			return "", err
		}
		vadStart := time.Now()
		p, err := e.speech(ctx, x, &v)
		m.AddDuration(metrics.VAD, time.Since(vadStart))
		if err != nil {
			return "", err
		}
		end := min(pos+512, w.Samples)
		if p >= float32(e.c.VADThreshold) {
			if start < 0 {
				start = max(int64(0), pos-pad)
			}
			lastVoice = end
			silenceAt = -1
		} else if p < float32(max(0.01, e.c.VADThreshold-0.15)) && start >= 0 && silenceAt < 0 {
			silenceAt = pos
		}
		if start < 0 {
			continue
		}
		if silenceAt >= 0 && end-silenceAt >= silence {
			stop := min(lastVoice+pad, end)
			if err := emit(start, stop); err != nil {
				return "", err
			}
			start = -1
			silenceAt = -1
			continue
		}
		if end-start >= chunk {
			stop := start + chunk
			var cutErr error
			stop, cutErr = quietBoundary(w, start, stop)
			if cutErr != nil {
				return "", cutErr
			}
			if err := emit(start, stop); err != nil {
				return "", err
			}
			start = stop - overlap
		}
	}
	if start >= 0 {
		if err := emit(start, w.Samples); err != nil {
			return "", err
		}
	}
	return result.String(), nil
}
func overlapSuffix(previous, next string, limits ...int) string {
	a, b := strings.Fields(previous), strings.Fields(next)
	normalize := func(s string) string {
		return strings.ToLower(strings.TrimFunc(s, func(r rune) bool { return unicode.IsPunct(r) }))
	}
	limit := min(len(a), len(b), 64)
	if len(limits) > 0 {
		limit = min(limit, limits[0])
	}
	for n := limit; n > 0; n-- {
		// An overlapped window can begin with a partial word. Match a suffix
		// inside its timed prefix and discard that already-covered fragment.
		for offset := limit - n; offset >= 0; offset-- {
			match := true
			for i := 0; i < n; i++ {
				if normalize(a[len(a)-n+i]) != normalize(b[offset+i]) {
					match = false
					break
				}
			}
			if match {
				return strings.Join(b[offset+n:], " ")
			}
		}
	}
	return next
}

// Prefer a quiet point near the window limit so a forced cut does not end in
// the middle of a word. The fallback still bounds genuinely continuous speech.
func quietBoundary(w *audio.WAV, start, stop int64) (int64, error) {
	from := max(start+(stop-start)/2, stop-3*audio.Rate)
	x, err := w.ReadSamples(from, int(stop-from))
	if err != nil {
		return 0, err
	}
	const width = 1600 // 100ms power window
	if len(x) < width {
		return stop, nil
	}
	var total float64
	for _, v := range x {
		total += float64(v) * float64(v)
	}
	if total == 0 {
		return stop, nil
	}
	var power float64
	for _, v := range x[:width] {
		power += float64(v) * float64(v)
	}
	best, bestAt := power, 0
	for i := 1; i+width <= len(x); i++ {
		old, new := float64(x[i-1]), float64(x[i+width-1])
		power += new*new - old*old
		if power <= best {
			best, bestAt = power, i
		}
	}
	if best/width < total/float64(len(x))*0.03 {
		return from + int64(bestAt+width/2), nil
	}
	return stop, nil
}
