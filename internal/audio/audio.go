package audio

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
)

var ErrInvalid = errors.New("invalid audio")
var ErrNoAudio = errors.New("no audio stream")
var ErrDuration = errors.New("duration exceeded")
var ErrUnavailable = errors.New("media tool unavailable")
var ErrStorage = errors.New("media storage unavailable")

const Rate = 16000
const formats = "wav,mp3,flac,mov,matroska,webm,ogg,aac"

type boundedBuffer struct {
	data  []byte
	limit int
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if left := b.limit - len(b.data); left > 0 {
		if len(p) > left {
			p = p[:left]
		}
		b.data = append(b.data, p...)
	}
	return n, nil
}
func command(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var out = boundedBuffer{limit: 64 << 10}
	var stderr = boundedBuffer{limit: 4096}
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	e := cmd.Run()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if e != nil {
		if errors.Is(e, exec.ErrNotFound) {
			return nil, ErrUnavailable
		}
		if strings.Contains(string(stderr.data), "No space left on device") {
			return nil, ErrStorage
		}
		return nil, ErrInvalid
	}
	return out.data, nil
}
func Probe(ctx context.Context, path string, max time.Duration) error {
	data, e := command(ctx, "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", formats, "-select_streams", "a:0", "-show_entries", "stream=codec_type,duration:format=duration", "-of", "json", path)
	if e != nil {
		return e
	}
	var p struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
			Duration  string `json:"duration"`
		} `json:"streams"`
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if json.Unmarshal(data, &p) != nil {
		return ErrInvalid
	}
	if len(p.Streams) == 0 {
		return ErrNoAudio
	}
	for _, v := range []string{p.Format.Duration, p.Streams[0].Duration} {
		if d, e := strconv.ParseFloat(v, 64); e == nil && !math.IsNaN(d) && d > max.Seconds() {
			return ErrDuration
		}
	}
	return nil
}
func Prepare(ctx context.Context, dir string, max time.Duration) (*WAV, error) {
	src := filepath.Join(dir, "source.bin")
	finishProbe := metrics.Start(ctx, metrics.Probe)
	probeErr := Probe(ctx, src, max)
	finishProbe(metrics.Code(probeErr))
	if e := probeErr; e != nil {
		return nil, e
	}
	dst := filepath.Join(dir, "prepared.wav")
	finishConvert := metrics.Start(ctx, metrics.Convert)
	_, e := command(ctx, "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-format_whitelist", formats, "-i", src, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-t", fmt.Sprintf("%.6f", max.Seconds()+1), "-rf64", "never", dst)
	finishConvert(metrics.Code(e))
	if e != nil {
		return nil, e
	}
	w, e := OpenWAV(dst)
	if e != nil {
		return nil, e
	}
	if float64(w.Samples)/Rate > max.Seconds() {
		w.Close()
		return nil, ErrDuration
	}
	return w, nil
}

type WAV struct {
	file    *os.File
	offset  int64
	Samples int64
}

func OpenWAV(path string) (*WAV, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	ok := false
	defer func() {
		if !ok {
			f.Close()
		}
	}()
	var h [12]byte
	if _, e = io.ReadFull(f, h[:]); e != nil || string(h[:4]) != "RIFF" || string(h[8:]) != "WAVE" {
		return nil, ErrInvalid
	}
	st, e := f.Stat()
	if e != nil {
		return nil, e
	}
	valid := false
	for {
		var ch [8]byte
		if _, e = io.ReadFull(f, ch[:]); e != nil {
			return nil, ErrInvalid
		}
		n := int64(binary.LittleEndian.Uint32(ch[4:]))
		off, _ := f.Seek(0, io.SeekCurrent)
		if n > st.Size()-off {
			return nil, ErrInvalid
		}
		switch string(ch[:4]) {
		case "fmt ":
			if n < 16 {
				return nil, ErrInvalid
			}
			var b [16]byte
			if _, e = io.ReadFull(f, b[:]); e != nil {
				return nil, ErrInvalid
			}
			valid = binary.LittleEndian.Uint16(b[0:]) == 1 && binary.LittleEndian.Uint16(b[2:]) == 1 && binary.LittleEndian.Uint32(b[4:]) == Rate && binary.LittleEndian.Uint16(b[12:]) == 2 && binary.LittleEndian.Uint16(b[14:]) == 16
		case "data":
			if !valid || n%2 != 0 {
				return nil, ErrInvalid
			}
			ok = true
			return &WAV{f, off, n / 2}, nil
		}
		if _, e = f.Seek(off+n+n%2, io.SeekStart); e != nil {
			return nil, e
		}
	}
}
func (w *WAV) Close() error { return w.file.Close() }
func (w *WAV) ReadSamples(start int64, count int) ([]float32, error) {
	if start < 0 || start > w.Samples || count < 0 {
		return nil, ErrInvalid
	}
	if int64(count) > w.Samples-start {
		count = int(w.Samples - start)
	}
	b := make([]byte, count*2)
	if _, e := w.file.ReadAt(b, w.offset+start*2); e != nil {
		return nil, e
	}
	x := make([]float32, count)
	for i := range x {
		x[i] = float32(int16(binary.LittleEndian.Uint16(b[i*2:]))) / 32768
	}
	return x, nil
}
