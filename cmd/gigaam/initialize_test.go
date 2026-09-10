package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/api"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
)

func TestInitializationFailureIsDiagnosedWithoutPrivateData(t *testing.T) {
	var output bytes.Buffer
	original := logging.Log
	logging.Log = logging.New(&output)
	t.Cleanup(func() { logging.Log = original })
	dir := t.TempDir()
	c := config.Config{
		Model:       "gigaam-v3-e2e-rnnt",
		Precision:   "fp32",
		ModelDir:    filepath.Join(dir, "private-models"),
		WorkDir:     filepath.Join(dir, "private-audio"),
		Key:         "private-test-key",
		Concurrency: 1,
	}
	service := api.New(c)
	if engine := initialize(context.Background(), c, service); engine != nil {
		engine.Close()
		t.Fatal("missing offline model accepted")
	}
	var failure map[string]any
	for _, line := range bytes.Split(bytes.TrimSpace(output.Bytes()), []byte("\n")) {
		var event map[string]any
		if err := json.Unmarshal(line, &event); err != nil {
			t.Fatal(err)
		}
		if event["msg"] == "initialization failed" {
			failure = event
		}
	}
	if failure["code"] != "model_unavailable" || failure["reason_code"] != "model_files_unavailable" {
		t.Fatalf("missing diagnostic: %v", failure)
	}
	for _, private := range []string{dir, c.Key, "private-models", "private-audio"} {
		if strings.Contains(output.String(), private) {
			t.Fatal("private data in startup logs")
		}
	}
	request := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	request.Header.Set("Authorization", "Bearer "+c.Key)
	response := httptest.NewRecorder()
	service.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), "model_unavailable") {
		t.Fatalf("unexpected readiness response: %d %s", response.Code, response.Body.String())
	}
}
