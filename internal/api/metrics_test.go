package api

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
)

func TestRequestMetrics(t *testing.T) {
	for _, tc := range []struct {
		name, code string
		stages     []string
	}{
		{"success", "ok", []string{"upload", "probe", "convert", "inference", "cleanup"}},
		{"invalid media", "invalid_audio", []string{"upload", "probe", "cleanup"}},
		{"invalid model", "model_mismatch", []string{"upload", "cleanup"}},
		{"cancelled", "request_cancelled", []string{"upload", "probe", "convert", "inference", "cleanup"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var logs bytes.Buffer
			previous := logging.Log
			logging.Log = logging.New(&logs)
			t.Cleanup(func() { logging.Log = previous })
			s := testServer(t)
			data := wavBytes()
			fields := map[string]string{"model": s.c.Model}
			switch tc.name {
			case "invalid media":
				data = []byte("private broken recording")
			case "invalid model":
				fields["model"] = "private model parameter"
			case "cancelled":
				s.Ready(fakeEngine(func(context.Context, *audio.WAV) (string, error) { return "", context.Canceled }))
			}
			w := httptest.NewRecorder()
			s.Handler().ServeHTTP(w, request(t, s, fields, data))
			if tc.code == "ok" {
				if w.Code != http.StatusOK {
					t.Fatal(w.Code, w.Body)
				}
			} else if responseCode(t, w) != tc.code {
				t.Fatal(w.Code, w.Body)
			}
			emptyWork(t, s)
			for _, secret := range []string{"test-secret", "private.wav", "Привет", "private broken recording", "private model parameter", s.c.WorkDir} {
				if strings.Contains(logs.String(), secret) {
					t.Fatalf("private data leaked: %s", secret)
				}
			}
			dec := json.NewDecoder(&logs)
			stages := make(map[string]int)
			var summary map[string]any
			for dec.More() {
				var row map[string]any
				if err := dec.Decode(&row); err != nil {
					t.Fatal(err)
				}
				if row["level"] != "INFO" || row["msg"] != "transcription" {
					t.Fatal("log schema changed", row)
				}
				stamp, ok := row["time"].(string)
				if !ok {
					t.Fatal("missing timestamp", row)
				}
				if _, err := time.Parse(time.RFC3339Nano, stamp); err != nil {
					t.Fatal(err)
				}
				if row["request_id"] != w.Header().Get("X-Request-ID") {
					t.Fatal("request correlation lost", row)
				}
				if row["stage"] == "complete" {
					summary = row
					continue
				}
				if row["event"] == "complete" {
					stages[row["stage"].(string)]++
					if d, ok := row["duration_ms"].(float64); !ok || d < 0 {
						t.Fatal(row)
					}
				}
			}
			if summary == nil || summary["code"] != tc.code {
				t.Fatal("missing final result", summary)
			}
			if len(stages) != len(tc.stages) {
				t.Fatal("unexpected stages", stages)
			}
			var stageTotal float64
			for _, stage := range tc.stages {
				if stages[stage] != 1 {
					t.Fatal("stage missing or duplicated", stage, stages)
				}
				d, ok := summary[stage+"_ms"].(float64)
				if !ok || d < 0 {
					t.Fatal("missing stage timing", summary)
				}
				stageTotal += d
			}
			if stageTotal > summary["duration_ms"].(float64) {
				t.Fatal("overlapping top-level timings", summary)
			}
			if summary["upload_bytes"] != float64(len(data)) {
				t.Fatal("incorrect upload size", summary)
			}
			if tc.code == "ok" {
				if summary["audio_seconds"] != float64(1) {
					t.Fatal("wrong audio duration", summary)
				}
				rtf, ok := summary["rtf"].(float64)
				if !ok || math.Abs(rtf-summary["duration_ms"].(float64)/1000) > 1e-12 {
					t.Fatal("wrong RTF", summary)
				}
			} else if _, ok := summary["rtf"]; ok {
				t.Fatal("failed requests must not report a completed RTF", summary)
			}
		})
	}
}
