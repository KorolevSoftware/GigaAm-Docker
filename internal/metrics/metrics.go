// Package metrics records request-local timings without retaining media or text.
package metrics

import (
	"context"
	"sync"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
	"github.com/rs/zerolog"
)

type Stage uint8

const (
	Upload Stage = iota
	Probe
	Convert
	Inference
	Cleanup
	VAD
	Frontend
	Encoder
	Decoder
	stageCount
)

var stageNames = [stageCount]string{"upload", "probe", "convert", "inference", "cleanup", "vad", "frontend", "encoder", "decoder"}

type timing struct {
	duration time.Duration
	recorded bool
}

type contextKey struct{}

type Request struct {
	mu                          sync.Mutex
	logger                      zerolog.Logger
	durations                   [stageCount]timing
	uploadBytes                 int64
	audioSeconds                float64
	asrChunks                   int
	asrAudioSeconds             float64
	hasUpload, hasAudio, hasASR bool
}

func NewContext(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, contextKey{}, &Request{
		logger: logging.Log.With().Str("request_id", id).Logger(),
	})
}

func FromContext(ctx context.Context) *Request {
	m, _ := ctx.Value(contextKey{}).(*Request)
	return m
}

func (m *Request) AddDuration(stage Stage, d time.Duration) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.durations[stage].duration += d
	m.durations[stage].recorded = true
}

func (m *Request) RecordUpload(bytes int64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.uploadBytes, m.hasUpload = bytes, true
}

func (m *Request) RecordAudio(seconds float64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.audioSeconds, m.hasAudio = seconds, true
}

func (m *Request) RecordASR(chunks int, seconds float64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.asrChunks, m.asrAudioSeconds, m.hasASR = chunks, seconds, true
}

// Start emits bounded stage events, not one log for every VAD frame or token.
func Start(ctx context.Context, stage Stage) func(string) {
	m := FromContext(ctx)
	if m == nil {
		return func(string) {}
	}
	name := stageNames[stage]
	m.logger.Info().Str("stage", name).Str("event", "start").Msg("transcription")
	start := time.Now()
	return func(code string) {
		d := time.Since(start)
		m.AddDuration(stage, d)
		m.logger.Info().Str("stage", name).Str("event", "complete").Float64("duration_ms", milliseconds(d)).Str("code", code).Msg("transcription")
	}
}

func Code(err error) string {
	if err != nil {
		return "error"
	}
	return "ok"
}

func milliseconds(d time.Duration) float64 { return float64(d) / float64(time.Millisecond) }

// Complete runs after temporary-file cleanup, before writing the HTTP response.
// Inference includes the VAD/frontend/encoder/decoder times; do not sum both.
func Complete(ctx context.Context, start time.Time, code string) {
	m := FromContext(ctx)
	if m == nil {
		return
	}
	total := time.Since(start)
	m.mu.Lock()
	defer m.mu.Unlock()
	event := m.logger.Info().Str("stage", "complete").Float64("duration_ms", milliseconds(total)).Str("code", code)
	for stage, value := range m.durations {
		if value.recorded {
			event.Float64(stageNames[stage]+"_ms", milliseconds(value.duration))
		}
	}
	if m.hasUpload {
		event.Int64("upload_bytes", m.uploadBytes)
	}
	if m.hasAudio {
		event.Float64("audio_seconds", m.audioSeconds)
	}
	if m.hasASR {
		event.Int("asr_chunks", m.asrChunks).Float64("asr_audio_seconds", m.asrAudioSeconds)
	}
	if m.hasAudio && m.audioSeconds > 0 && code == "ok" {
		event.Float64("rtf", total.Seconds()/m.audioSeconds)
		if inference := m.durations[Inference].duration; inference > 0 {
			event.Float64("inference_rtf", inference.Seconds()/m.audioSeconds)
		}
	}
	event.Msg("transcription")
}
