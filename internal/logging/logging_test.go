package logging

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"testing"
)

func TestConcurrentRequestLoggers(t *testing.T) {
	var output bytes.Buffer
	logger := New(&output)
	var workers sync.WaitGroup
	for i := 0; i < 8; i++ {
		workers.Go(func() {
			child := logger.With().Str("request_id", fmt.Sprint(i)).Logger()
			for n := 0; n < 25; n++ {
				child.Info().Int("sequence", n).Msg("event")
			}
		})
	}
	workers.Wait()
	seen := make(map[string]bool)
	decoder := json.NewDecoder(&output)
	for {
		var row struct {
			Request  string `json:"request_id"`
			Sequence int    `json:"sequence"`
			Level    string `json:"level"`
			Message  string `json:"msg"`
		}
		if err := decoder.Decode(&row); err == io.EOF {
			break
		} else if err != nil {
			t.Fatal("interleaved JSON", err)
		}
		if row.Level != "INFO" || row.Message != "event" {
			t.Fatal(row)
		}
		key := fmt.Sprintf("%s/%d", row.Request, row.Sequence)
		if seen[key] {
			t.Fatal("duplicate event", key)
		}
		seen[key] = true
	}
	if len(seen) != 200 {
		t.Fatal("lost events", len(seen))
	}
}
