package inference

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/models"
	ort "github.com/yalue/onnxruntime_go"
)

func TestFeatures(t *testing.T) {
	f := NewFeatures()
	x, n := f.Compute(make([]float32, 16000))
	if n != 99 || len(x) != 64*99 {
		t.Fatal(n, len(x))
	}
	for _, v := range x {
		if math.Abs(float64(v)-math.Log(1e-9)) > 1e-5 {
			t.Fatal(v)
		}
	}
	x, n = f.Compute([]float32{1})
	if n != 1 || len(x) != 64 {
		t.Fatal("short audio")
	}
}
func TestOverlap(t *testing.T) {
	for _, c := range [][3]string{{"Это очень длинная запись", "длинная запись продолжается.", "продолжается."}, {"Привет, мир!", "мир сегодня.", "сегодня."}, {"Один два", "три четыре", "три четыре"}, {"да", "да да", "да"}} {
		if got := overlapSuffix(c[0], c[1]); got != c[2] {
			t.Fatalf("%q != %q", got, c[2])
		}
	}
}
func TestSentencePieceDecode(t *testing.T) {
	e := &Engine{vocab: []string{"<unk>", "▁Привет", ",", "▁мир", "!", "<blk>"}, blank: 5}
	if s := e.decode([]int{1, 2, 3, 4}); s != "Привет, мир!" {
		t.Fatal(s)
	}
}
func TestNativeVAD(t *testing.T) {
	lib := os.Getenv("GIGAAM_TEST_ORT")
	path := os.Getenv("GIGAAM_TEST_VAD")
	if lib == "" || path == "" {
		t.Skip("set GIGAAM_TEST_ORT and GIGAAM_TEST_VAD for native test")
	}
	ort.SetSharedLibraryPath(lib)
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatal(err)
	}
	defer ort.DestroyEnvironment()
	g, err := openGraph(path, []string{"input", "state", "sr"}, []string{"output", "stateN"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer g.close()
	e := &Engine{vad: g}
	v := vadState{state: make([]float32, 256)}
	for i := 0; i < 10; i++ {
		p, err := e.speech(context.Background(), make([]float32, 512), &v)
		if err != nil || p > 0.5 {
			t.Fatal(p, err)
		}
	}
}

// This opt-in test runs both full FP32 graphs against release golden fixtures.
// Reference JSON is generated only by scripts/reference.py, never by Go.
func TestGolden(t *testing.T) {
	root, lib, fixtures := os.Getenv("GIGAAM_TEST_MODELS"), os.Getenv("GIGAAM_TEST_ORT"), os.Getenv("GIGAAM_TEST_GOLDEN")
	if root == "" || lib == "" || fixtures == "" {
		t.Skip("set GIGAAM_TEST_MODELS, GIGAAM_TEST_ORT, GIGAAM_TEST_GOLDEN")
	}
	m, err := models.ManifestData()
	if err != nil {
		t.Fatal(err)
	}
	dirs := map[string]string{}
	for _, b := range m.Bundles {
		dirs[b.ID] = models.Directory(root, b)
	}
	for _, name := range []string{"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"} {
		t.Run(name, func(t *testing.T) {
			c := config.Config{Model: name, ORTLibrary: lib, Threads: 2, Chunk: 20 * time.Second, Overlap: time.Second, Padding: 300 * time.Millisecond, MinSilence: 500 * time.Millisecond, VADThreshold: 0.5}
			e, err := New(context.Background(), c, dirs[name], dirs["silero-vad"])
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			files, err := filepath.Glob(filepath.Join(fixtures, name, "*.json"))
			if err != nil || len(files) == 0 {
				t.Fatal("no golden fixtures", err)
			}
			for _, file := range files {
				b, err := os.ReadFile(file)
				if err != nil {
					t.Fatal(err)
				}
				var ref struct {
					WAV      string    `json:"wav"`
					Text     string    `json:"text"`
					Features []float32 `json:"features"`
				}
				if err = json.Unmarshal(b, &ref); err != nil {
					t.Fatal(err)
				}
				w, err := audio.OpenWAV(filepath.Join(fixtures, ref.WAV))
				if err != nil {
					t.Fatal(err)
				}
				samples, err := w.ReadSamples(0, int(w.Samples))
				w.Close()
				if err != nil {
					t.Fatal(err)
				}
				f, _ := e.features.Compute(samples)
				if len(f) != len(ref.Features) {
					t.Fatal("feature shape mismatch")
				}
				var maxErr float64
				for i, v := range f {
					maxErr = math.Max(maxErr, math.Abs(float64(v-ref.Features[i])))
				}
				for i, v := range f {
					if !featureClose(v, ref.Features[i]) {
						t.Fatalf("features mismatch at %d; max log error %.6g", i, maxErr)
					}
				}
				got, err := e.recognize(context.Background(), samples)
				if err != nil {
					t.Fatal(err)
				}
				if got != ref.Text {
					t.Fatalf("transcript mismatch for %s", filepath.Base(file))
				}
				t.Logf("%s: feature max abs error %.6g; transcript matches", filepath.Base(file), maxErr)
			}
		})
	}
}

// Smoke test real graphs before generating golden references. The WAV must be
// a public/test recording: unlike the server, this test may print its text.
func TestNativeSmoke(t *testing.T) {
	root, lib, wavPath := os.Getenv("GIGAAM_TEST_MODELS"), os.Getenv("GIGAAM_TEST_ORT"), os.Getenv("GIGAAM_TEST_WAV")
	if root == "" || lib == "" || wavPath == "" {
		t.Skip("set native smoke test paths")
	}
	m, _ := models.ManifestData()
	dirs := map[string]string{}
	for _, b := range m.Bundles {
		dirs[b.ID] = models.Directory(root, b)
	}
	names := []string{"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"}
	if n := os.Getenv("GIGAAM_TEST_MODEL"); n != "" {
		names = []string{n}
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			c := config.Config{Model: name, ORTLibrary: lib, Threads: 2, Chunk: 20 * time.Second, Overlap: time.Second, Padding: 300 * time.Millisecond, MinSilence: 500 * time.Millisecond, VADThreshold: 0.5}
			e, err := New(context.Background(), c, dirs[name], dirs["silero-vad"])
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			w, err := audio.OpenWAV(wavPath)
			if err != nil {
				t.Fatal(err)
			}
			defer w.Close()
			start := time.Now()
			text, err := e.Transcribe(context.Background(), w)
			if err != nil {
				t.Fatal(err)
			}
			if text == "" {
				t.Fatal("speech fixture yielded empty text")
			}
			if expectedPath := os.Getenv("GIGAAM_TEST_EXPECTED_TEXT_FILE"); expectedPath != "" {
				expected, err := os.ReadFile(expectedPath)
				if err != nil {
					t.Fatal(err)
				}
				if strings.TrimSpace(text) != strings.TrimSpace(string(expected)) {
					t.Fatal("long-audio text differs from expected fixture")
				}
			}
			t.Logf("duration=%v text=%s", time.Since(start), text)
			for _, seconds := range []int{1, 20, 30} {
				if _, err = e.recognize(context.Background(), make([]float32, seconds*audio.Rate)); err != nil {
					t.Fatalf("window %ds: %v", seconds, err)
				}
			}
		})
	}
}

func TestOverlapCannotRemoveRepeatedSentencesOutsideIntersection(t *testing.T) {
	prev := "Здравствуйте. Сегодня проверяем речь. Здравствуйте. Сегодня проверяем речь."
	next := "Здравствуйте. Сегодня проверяем речь. И продолжаем."
	if got := overlapSuffix(prev, next, 2); got != next {
		t.Fatal("removed words outside the audio overlap:", got)
	}
}

func TestFrontendAgainstTorchFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/features.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		PCM      []int16   `json:"pcm"`
		Features []float32 `json:"features"`
	}
	if err = json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	samples := make([]float32, len(ref.PCM))
	for i, v := range ref.PCM {
		samples[i] = float32(v) / 32768
	}
	actual, _ := NewFeatures().Compute(samples)
	if len(actual) != len(ref.Features) {
		t.Fatal("shape mismatch")
	}
	var delta float64
	worst := 0
	for i, v := range actual {
		if math.Abs(float64(v-ref.Features[i])) > delta {
			worst = i
		}
		delta = math.Max(delta, math.Abs(float64(v-ref.Features[i])))
	}
	for i, v := range actual {
		if !featureClose(v, ref.Features[i]) {
			t.Fatalf("Torch frontend mismatch: max log error %.8f at %d", delta, worst)
		}
	}

}

// Float32 FFT cancellation amplifies relative error in near-silent bins. Keep
// the strict log tolerance above that floor; below it also bound absolute
// linear-mel error and cap the log difference at two percent.
func featureClose(a, b float32) bool {
	d := math.Abs(float64(a - b))
	if d <= 0.002 {
		return true
	}
	pa, pb := math.Exp(float64(a)), math.Exp(float64(b))
	return d <= 0.02 && math.Max(pa, pb) < 1e-6 && math.Abs(pa-pb) <= 1e-9
}

func TestNativeCancellation(t *testing.T) {
	root, lib := os.Getenv("GIGAAM_TEST_MODELS"), os.Getenv("GIGAAM_TEST_ORT")
	if root == "" || lib == "" {
		t.Skip("set native model and runtime paths")
	}
	m, _ := models.ManifestData()
	dirs := map[string]string{}
	for _, b := range m.Bundles {
		dirs[b.ID] = models.Directory(root, b)
	}
	for _, name := range []string{"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"} {
		t.Run(name, func(t *testing.T) {
			e, err := New(context.Background(), config.Config{Model: name, ORTLibrary: lib, Threads: 2}, dirs[name], dirs["silero-vad"])
			if err != nil {
				t.Fatal(err)
			}
			defer e.Close()
			features, n := e.features.Compute(make([]float32, 30*audio.Rate))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			timer := time.AfterFunc(10*time.Millisecond, cancel)
			defer timer.Stop()
			start := time.Now()
			_, err = e.encoder.run(ctx, ft([]int64{1, 64, int64(n)}, features), it([]int64{1}, []int64{int64(n)}))
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("expected cancellation, got %v", err)
			}
			t.Logf("30-second encoder call returned after cancellation in %v", time.Since(start))
		})
	}
}
