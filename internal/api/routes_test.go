package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRoutes(t *testing.T) {
	server := testServer(t)
	handler := server.Handler()
	for _, tc := range []struct {
		name       string
		method     string
		path       string
		authorized bool
		status     int
		code       string
		allow      string
	}{
		{name: "health without key", method: http.MethodGet, path: "/healthz", status: http.StatusOK},
		{name: "health HEAD", method: http.MethodHead, path: "/healthz", status: http.StatusOK},
		{name: "health unknown suffix", method: http.MethodGet, path: "/healthz/extra", status: http.StatusNotFound},
		{name: "unknown public route", method: http.MethodGet, path: "/unknown", status: http.StatusNotFound},
		{name: "models", method: http.MethodGet, path: "/v1/models", authorized: true, status: http.StatusOK},
		{name: "models method", method: http.MethodPost, path: "/v1/models", authorized: true, status: http.StatusMethodNotAllowed, allow: "GET, HEAD"},
		{name: "models HEAD", method: http.MethodHead, path: "/v1/models", authorized: true, status: http.StatusOK},
		{name: "transcription method", method: http.MethodGet, path: "/v1/audio/transcriptions", authorized: true, status: http.StatusMethodNotAllowed, allow: http.MethodPost},
		{name: "transcription OPTIONS", method: http.MethodOptions, path: "/v1/audio/transcriptions", authorized: true, status: http.StatusMethodNotAllowed, allow: http.MethodPost},
		{name: "unknown API route", method: http.MethodGet, path: "/v1/unknown", authorized: true, status: http.StatusNotFound},
		{name: "models suffix", method: http.MethodGet, path: "/v1/models/extra", authorized: true, status: http.StatusNotFound},
		{name: "authenticate before 404", method: http.MethodGet, path: "/v1/unknown", status: http.StatusUnauthorized, code: "invalid_api_key"},
		{name: "authenticate before 405", method: http.MethodDelete, path: "/v1/models", status: http.StatusUnauthorized, code: "invalid_api_key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(tc.method, tc.path, nil)
			if tc.authorized {
				request.Header.Set("Authorization", "Bearer test-secret")
			}
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			if response.Code != tc.status {
				t.Fatalf("status = %d, want %d: %s", response.Code, tc.status, response.Body)
			}
			if tc.code != "" && responseCode(t, response) != tc.code {
				t.Fatalf("unexpected error: %s", response.Body)
			}
			if response.Header().Get("Allow") != tc.allow {
				t.Errorf("Allow = %q, want %q", response.Header().Get("Allow"), tc.allow)
			}
			if tc.code != "" && response.Header().Get("Content-Type") != "application/json" {
				t.Error("error is not JSON")
			}
			if tc.status == http.StatusNotFound || tc.status == http.StatusMethodNotAllowed {
				if response.Header().Get("Content-Type") != "text/plain; charset=utf-8" {
					t.Error("expected standard ServeMux error response")
				}
			}
			if tc.authorized || tc.status == http.StatusUnauthorized {
				if len(response.Header().Get("X-Request-ID")) != 32 || response.Header().Get("Cache-Control") != "no-store" {
					t.Error("missing API middleware headers")
				}
			}
			emptyWork(t, server)
		})
	}
}
