package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
)

type fakeEngine func(context.Context, *audio.WAV) (string, error)

func (f fakeEngine) Transcribe(c context.Context, w *audio.WAV) (string, error) { return f(c, w) }
func testServer(t *testing.T) *Server {
	t.Helper()
	c := config.Config{Key: "test-secret", Model: "gigaam-v3-e2e-rnnt", WorkDir: t.TempDir(), Concurrency: 1, MaxFile: 1 << 20, ReserveBytes: 1, UploadTimeout: time.Second, ProcessTimeout: 5 * time.Second, ResponseTimeout: time.Second}
	s := New(c)
	s.freeBytes = func(string) (uint64, error) { return 1 << 40, nil }
	s.Ready(fakeEngine(func(context.Context, *audio.WAV) (string, error) { return "Привет, мир!", nil }))
	return s
}
func wavBytes() []byte {
	var b bytes.Buffer
	b.WriteString("RIFF")
	_ = binary.Write(&b, binary.LittleEndian, uint32(36+32000))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), uint16(1), uint16(1), uint32(16000), uint32(32000), uint16(2), uint16(16)} {
		_ = binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	_ = binary.Write(&b, binary.LittleEndian, uint32(32000))
	b.Write(make([]byte, 32000))
	return b.Bytes()
}
func request(t *testing.T, s *Server, fields map[string]string, data []byte) *http.Request {
	t.Helper()
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	if data != nil {
		p, _ := w.CreateFormFile("file", "../../private.wav")
		_, _ = p.Write(data)
	}
	if fields == nil {
		fields = map[string]string{"model": s.c.Model}
	}
	for k, v := range fields {
		_ = w.WriteField(k, v)
	}
	_ = w.Close()
	r := httptest.NewRequest("POST", "/v1/audio/transcriptions", &b)
	r.Header.Set("Content-Type", w.FormDataContentType())
	r.Header.Set("Authorization", "Bearer test-secret")
	return r
}
func responseCode(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var v struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if e := json.Unmarshal(w.Body.Bytes(), &v); e != nil {
		t.Fatal(e)
	}
	return v.Error.Code
}
func emptyWork(t *testing.T, s *Server) {
	t.Helper()
	files, e := os.ReadDir(s.c.WorkDir)
	if e != nil || len(files) != 0 {
		t.Fatalf("temporary files remain: %v %v", files, e)
	}
}
func TestSuccessAndCleanupBeforeWrite(t *testing.T) {
	for _, format := range []string{"json", "text"} {
		t.Run(format, func(t *testing.T) {
			s := testServer(t)
			w := &cleanupWriter{ResponseRecorder: httptest.NewRecorder(), t: t, s: s}
			s.Handler().ServeHTTP(w, request(t, s, map[string]string{"model": s.c.Model, "response_format": format}, wavBytes()))
			if w.Code != http.StatusOK {
				t.Fatalf("%d %s", w.Code, w.Body)
			}
			if !strings.Contains(w.Body.String(), "Привет, мир!") {
				t.Fatal(w.Body)
			}
			if w.Header().Get("Cache-Control") != "no-store" || len(w.Header().Get("X-Request-ID")) != 32 {
				t.Fatal(w.Header())
			}
			emptyWork(t, s)
		})
	}
}

type cleanupWriter struct {
	*httptest.ResponseRecorder
	t *testing.T
	s *Server
}

func (w *cleanupWriter) WriteHeader(n int) { emptyWork(w.t, w.s); w.ResponseRecorder.WriteHeader(n) }
func TestValidation(t *testing.T) {
	cases := []struct {
		name   string
		fields map[string]string
		data   []byte
		code   string
	}{
		{"missing file", map[string]string{"model": "gigaam-v3-e2e-rnnt"}, nil, "missing_file"},
		{"missing model", map[string]string{}, wavBytes(), "missing_model"},
		{"model", map[string]string{"model": "other"}, wavBytes(), "model_mismatch"},
		{"language", map[string]string{"model": "gigaam-v3-e2e-rnnt", "language": "en"}, wavBytes(), "unsupported_language"},
		{"stream", map[string]string{"model": "gigaam-v3-e2e-rnnt", "stream": "true"}, wavBytes(), "unsupported_stream"},
		{"format", map[string]string{"model": "gigaam-v3-e2e-rnnt", "response_format": "verbose_json"}, wavBytes(), "unsupported_response_format"},
		{"prompt", map[string]string{"model": "gigaam-v3-e2e-rnnt", "prompt": "secret"}, wavBytes(), "unsupported_parameter"},
		{"empty", nil, []byte{}, "empty_file"},
		{"corrupt", nil, []byte("invalid media"), "invalid_audio"},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s := testServer(t)
			w := httptest.NewRecorder()
			s.Handler().ServeHTTP(w, request(t, s, tt.fields, tt.data))
			if got := responseCode(t, w); got != tt.code {
				t.Fatalf("got %s: %s", got, w.Body)
			}
			emptyWork(t, s)
		})
	}
}
func TestFileSize(t *testing.T) {
	s := testServer(t)
	s.c.MaxFile = 10
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, request(t, s, nil, wavBytes()))
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatal(w.Code, w.Body)
	}
	emptyWork(t, s)
}
func TestHealthIndependent(t *testing.T) {
	s := testServer(t)
	s.Unavailable()
	s.freeBytes = func(string) (uint64, error) { return 0, errors.New("disk") }
	s.pending["path"] = true
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/healthz", nil))
	if w.Code != http.StatusOK || w.Body.String() != "{\"status\":\"ok\"}\n" {
		t.Fatal(w.Code, w.Body)
	}
}
func TestAuthAndReadiness(t *testing.T) {
	s := testServer(t)
	for _, state := range []string{"service_initializing", "model_unavailable", ""} {
		s.state = state
		for _, path := range []string{"/v1/models", "/v1/audio/transcriptions"} {
			r := httptest.NewRequest("GET", path, nil)
			w := httptest.NewRecorder()
			s.Handler().ServeHTTP(w, r)
			if w.Code != http.StatusUnauthorized {
				t.Fatal(w.Code)
			}
			r = httptest.NewRequest("GET", "/v1/models", nil)
			r.Header.Set("Authorization", "Bearer test-secret")
			w = httptest.NewRecorder()
			s.Handler().ServeHTTP(w, r)
			if state != "" {
				if responseCode(t, w) != state {
					t.Fatal(w.Body)
				}
			} else if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), s.c.Model) {
				t.Fatal(w.Body)
			}
		}
	}
}
func TestCleanupFailureBlocksAndRecovers(t *testing.T) {
	s := testServer(t)
	s.removeAll = func(string) error { return errors.New("permission denied") }
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, request(t, s, nil, wavBytes()))
	if responseCode(t, w) != "cleanup_failed" {
		t.Fatal(w.Body)
	}
	w = httptest.NewRecorder()
	s.Handler().ServeHTTP(w, request(t, s, nil, wavBytes()))
	if responseCode(t, w) != "cleanup_pending" {
		t.Fatal(w.Body)
	}
	s.removeAll = os.RemoveAll
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.RetryCleanup(ctx)
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		n := len(s.pending)
		s.mu.Unlock()
		if n == 0 {
			emptyWork(t, s)
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("cleanup not retried")
}
func TestBusyAndCancellationHoldSlot(t *testing.T) {
	s := testServer(t)
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	s.Ready(fakeEngine(func(ctx context.Context, w *audio.WAV) (string, error) {
		close(entered)
		<-release
		return "partial", ctx.Err()
	}))
	r := request(t, s, nil, wavBytes())
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	r = r.WithContext(ctx)
	go func() { defer close(done); s.Handler().ServeHTTP(httptest.NewRecorder(), r) }()
	<-entered
	cancel()
	w := httptest.NewRecorder()
	r2 := request(t, s, nil, wavBytes())
	body := &countReader{ReadCloser: r2.Body}
	r2.Body = body
	s.Handler().ServeHTTP(w, r2)
	if w.Code != http.StatusTooManyRequests || body.n != 0 || w.Header().Get("Retry-After") == "" {
		t.Fatal(w.Code, body.n)
	}
	close(release)
	<-done
	emptyWork(t, s)
}

type countReader struct {
	io.ReadCloser
	n int
}

func (c *countReader) Read(p []byte) (int, error) {
	n, e := c.ReadCloser.Read(p)
	c.n += n
	return n, e
}
func TestProcessingTimeout(t *testing.T) {
	s := testServer(t)
	s.c.ProcessTimeout = 100 * time.Millisecond
	s.Ready(fakeEngine(func(ctx context.Context, w *audio.WAV) (string, error) { <-ctx.Done(); return "partial", ctx.Err() }))
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, request(t, s, nil, wavBytes()))
	if w.Code != http.StatusGatewayTimeout {
		t.Fatal(w.Code, w.Body)
	}
	emptyWork(t, s)
}
func TestStorageAndStartupCleanup(t *testing.T) {
	s := testServer(t)
	p := filepath.Join(s.c.WorkDir, "request-old")
	_ = os.Mkdir(p, 0700)
	_ = os.WriteFile(filepath.Join(p, "source.bin"), []byte("secret"), 0600)
	if e := s.CleanStartup(); e != nil {
		t.Fatal(e)
	}
	emptyWork(t, s)
	s.freeBytes = func(string) (uint64, error) { return 0, nil }
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, request(t, s, nil, wavBytes()))
	if w.Code != http.StatusServiceUnavailable || responseCode(t, w) != "storage_unavailable" {
		t.Fatal(w.Code, w.Body)
	}
	emptyWork(t, s)
}
func TestDrainCancelsThenWaitsForNative(t *testing.T) {
	s := testServer(t)
	entered, finish := make(chan struct{}), make(chan struct{})
	var wg sync.WaitGroup
	s.Ready(fakeEngine(func(ctx context.Context, w *audio.WAV) (string, error) {
		close(entered)
		<-ctx.Done()
		<-finish
		return "", ctx.Err()
	}))
	wg.Add(1)
	go func() { defer wg.Done(); s.Handler().ServeHTTP(httptest.NewRecorder(), request(t, s, nil, wavBytes())) }()
	<-entered
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	drained := make(chan struct{})
	go func() { s.Drain(ctx); close(drained) }()
	<-ctx.Done()
	select {
	case <-drained:
		t.Fatal("released native call early")
	default:
	}
	w := httptest.NewRecorder()
	s.Handler().ServeHTTP(w, httptest.NewRequest("GET", "/healthz", nil))
	if w.Code != http.StatusOK {
		t.Fatal(w.Code)
	}
	close(finish)
	<-drained
	wg.Wait()
	emptyWork(t, s)
}

func TestLiveSlowUploadAndEarlyRejection(t *testing.T) {
	for _, kind := range []string{"unauthorized", "busy", "timeout"} {
		t.Run(kind, func(t *testing.T) {
			s := testServer(t)
			s.c.UploadTimeout = 50 * time.Millisecond
			if kind == "busy" {
				_, e := s.acquire()
				if e != nil {
					t.Fatal(e)
				}
				defer s.release()
			}
			ts := httptest.NewServer(s.Handler())
			defer ts.Close()
			conn, e := net.Dial("tcp", strings.TrimPrefix(ts.URL, "http://"))
			if e != nil {
				t.Fatal(e)
			}
			defer conn.Close()
			_ = conn.SetDeadline(time.Now().Add(2 * time.Second))
			auth := "Bearer test-secret"
			if kind == "unauthorized" {
				auth = "invalid"
			}
			partial := "--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.wav\"\r\n\r\npartial"
			_, e = fmt.Fprintf(conn, "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: test\r\nAuthorization: %s\r\nContent-Type: multipart/form-data; boundary=test\r\nContent-Length: 100000\r\n\r\n%s", auth, partial)
			if e != nil {
				t.Fatal(e)
			}
			response, e := http.ReadResponse(bufio.NewReader(conn), nil)
			if e != nil {
				t.Fatal("response waited for unread body:", e)
			}
			defer response.Body.Close()
			_, _ = io.ReadAll(response.Body)
			want := map[string]int{"unauthorized": http.StatusUnauthorized, "busy": http.StatusTooManyRequests, "timeout": http.StatusRequestTimeout}[kind]
			if response.StatusCode != want {
				t.Fatal(response.StatusCode, want)
			}
			emptyWork(t, s)
		})
	}
}
