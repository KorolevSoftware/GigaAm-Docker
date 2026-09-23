package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/GigaAM-CTC/audio"
	airuntime "github.com/KorolevSoftware/GigaAm-Docker/GigaAM-CTC/internal/inference"
	"golang.org/x/sync/errgroup"
)

// envOr возвращает переменную окружения или значение по умолчанию.
func envOr(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func fail(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}

func main() {
	ctcModelPath := os.Getenv("CTC_MODE_PATH")
	vadModelPath := os.Getenv("VAD_MODEL_PATH")
	coefficientsPath := os.Getenv("FEATURE_COEFFICIENTS_PATH")
	if coefficientsPath == "" {
		fail("FEATURE_COEFFICIENTS_PATH is not set")
	}
	// Словарь по умолчанию лежит рядом с моделью CTC.
	vocabPath := envOr("CTC_VOCAB_PATH", filepath.Join(filepath.Dir(ctcModelPath), "v3_e2e_ctc_vocab.txt"))
	audioPath := envOr("AUDIO_PATH", "test/test_datka.wav")

	file, err := os.Open(audioPath)
	if err != nil {
		fail("open audio: %v", err)
	}
	defer file.Close()

	wav := audio.Load(file)
	if wav == nil {
		fail("audio %s is not a valid wav", audioPath)
	}
	audioRaw, err := wav.Read()
	if err != nil {
		fail("read audio: %v", err)
	}

	if err := airuntime.Load(ctcModelPath, vadModelPath); err != nil {
		fail("init onnxruntime: %v", err)
	}

	vad, err := airuntime.MakeVadSession(vadModelPath)
	if err != nil {
		fail("load vad: %v", err)
	}
	provider := envOr("CTC_PROVIDER", "cpu")
	loadStarted := time.Now()
	ctc, err := airuntime.MakeCTCSession(ctcModelPath, coefficientsPath, vocabPath, provider)
	if err != nil {
		fail("load ctc: %v", err)
	}

	loadSeconds := time.Since(loadStarted).Seconds()

	started := time.Now()

	chanVad := make(chan float32, 10)

	g, ctx := errgroup.WithContext(context.Background()) // группа + контекст, который отменится при первой ошибке

	// VAD идёт параллельно с CTC, поэтому его время меряем внутри горутины.
	// Читать vadSeconds безопасно только после g.Wait().
	var vadSeconds float64
	g.Go(func() error {
		defer close(chanVad)
		t := time.Now()
		err := vad.Run(ctx, audioRaw, chanVad)
		vadSeconds = time.Since(t).Seconds()
		return err
	})

	windows := airuntime.MakeWindows(ctx, chanVad, audioRaw, 20)

	var transcript string
	var segments []airuntime.Segment
	g.Go(func() error {
		var err error
		transcript, segments, err = ctc.Run(ctx, windows, audioRaw)
		return err
	})

	if err := g.Wait(); err != nil {
		fail("pipeline: %v", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	for i, s := range segments {
		enc.Encode(map[string]any{
			"event": "segment", "window": i + 1,
			"start_s": float64(s.Start) / airuntime.RATE, "stop_s": float64(s.Stop) / airuntime.RATE,
			"text": s.Text, "addition": s.Addition,
		})
	}
	enc.Encode(map[string]any{
		"event": "done", "windows": len(segments), "transcript": transcript,
		"provider": provider, "load_s": loadSeconds, "vad_s": vadSeconds, "total_s": time.Since(started).Seconds(),
	})
}
