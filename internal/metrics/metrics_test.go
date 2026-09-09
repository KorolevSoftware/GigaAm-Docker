package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
)

func TestSummaryPreservesZerosAndAggregatesConcurrentTimings(t *testing.T) {
	var output bytes.Buffer
	original := logging.Log
	logging.Log = logging.New(&output)
	t.Cleanup(func() { logging.Log = original })
	ctx := NewContext(context.Background(), "test-request")
	m := FromContext(ctx)
	m.RecordUpload(0)
	m.RecordAudio(0)
	m.RecordASR(0, 0)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 100 {
				m.AddDuration(Encoder, 125*time.Microsecond)
			}
		})
	}
	wg.Wait()
	Complete(ctx, time.Now(), "ok")
	var result map[string]json.RawMessage
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"upload_bytes", "audio_seconds", "asr_chunks", "asr_audio_seconds"} {
		if string(result[field]) != "0" {
			t.Errorf("%s = %s, want explicit zero", field, result[field])
		}
	}
	var encoderMS float64
	if err := json.Unmarshal(result["encoder_ms"], &encoderMS); err != nil {
		t.Fatal(err)
	}
	if encoderMS != 100 {
		t.Fatalf("encoder_ms = %v, want 100", encoderMS)
	}
	for _, field := range []string{"upload_ms", "vad_ms", "rtf", "inference_rtf"} {
		if _, ok := result[field]; ok {
			t.Errorf("unexpected %s", field)
		}
	}
	output.Reset()
	Complete(NewContext(context.Background(), "empty"), time.Now(), "error")
	result = nil
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"upload_bytes", "audio_seconds", "asr_chunks", "asr_audio_seconds"} {
		if _, ok := result[field]; ok {
			t.Errorf("unrecorded %s present", field)
		}
	}
}
