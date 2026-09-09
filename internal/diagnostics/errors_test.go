package diagnostics

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"testing"
)

func TestCodeAndCause(t *testing.T) {
	cause := &url.Error{Op: "Get", URL: "https://example.invalid/private?token=secret", Err: errors.New("connection failed")}
	err := fmt.Errorf("model setup: %w", Wrap("download_network_failed", cause))
	if Code(err) != "download_network_failed" {
		t.Fatal(Code(err))
	}
	if !errors.Is(err, cause) {
		t.Fatal("original error lost")
	}
	var networkError *url.Error
	if !errors.As(err, &networkError) || networkError != cause {
		t.Fatal("network error details lost")
	}
	for _, tc := range []struct {
		err  error
		code string
	}{
		{nil, "ok"},
		{Wrap("download_network_failed", context.DeadlineExceeded), "timeout"},
		{Wrap("asr_warmup_failed", context.Canceled), "canceled"},
	} {
		if got := Code(tc.err); got != tc.code {
			t.Errorf("code = %s, want %s", got, tc.code)
		}
	}
	if Wrap("unused", nil) != nil {
		t.Fatal("nil error wrapped")
	}
}
