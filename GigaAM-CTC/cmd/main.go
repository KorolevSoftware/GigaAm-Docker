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

	ctx := context.Background()
	started := time.Now()

	vadData, err := vad.Run(ctx, audioRaw)
	if err != nil {
		fail("vad: %v", err)
	}
	windows := airuntime.MakeWindows(vadData, audioRaw, 20)
	vadDone := time.Since(started)

	transcript, segments, err := ctc.Run(ctx, windows, audioRaw)
	if err != nil {
		fail("ctc: %v", err)
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
		"event": "done", "windows": len(windows), "transcript": transcript,
		"provider": provider, "load_s": loadSeconds, "vad_s": vadDone.Seconds(), "total_s": time.Since(started).Seconds(),
	})
}
