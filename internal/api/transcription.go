package api

import (
	"context"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
)

func (s *Server) transcribe(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	engine, e := s.acquire()
	if e != nil {
		if e.Status == http.StatusTooManyRequests {
			w.Header().Set("Retry-After", "5")
		}
		metrics.Complete(r.Context(), start, e.Code)
		writeError(w, e)
		return
	}
	defer s.release()
	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()
	stop := context.AfterFunc(s.root, cancel)
	defer stop()
	r = r.WithContext(ctx)
	rc := http.NewResponseController(w)
	stopRead := context.AfterFunc(ctx, func() { _ = rc.SetReadDeadline(time.Now()) })
	defer stopRead()
	_ = rc.SetReadDeadline(time.Now().Add(s.c.UploadTimeout))
	text, format, err := s.process(r, engine, rc)
	_ = rc.SetReadDeadline(time.Time{})
	_ = rc.SetWriteDeadline(time.Now().Add(s.c.ResponseTimeout))
	if ctx.Err() != nil && err == nil {
		err = problem(statusClientClosedRequest, "request_cancelled", "Request cancelled.", nil)
	}
	if err == nil {
		w.Header().Del("Connection")
	}
	code := "ok"
	if err != nil {
		code = err.Code
	}
	metrics.Complete(r.Context(), start, code)
	if err != nil {
		writeError(w, err)
		return
	}
	if format == "text" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, text)
	} else {
		writeJSON(w, http.StatusOK, map[string]string{"text": text})
	}
}

func (s *Server) process(r *http.Request, engine Engine, rc *http.ResponseController) (text, format string, err *Error) {
	if e := s.storage(s.c.MaxFile); e != nil {
		return "", "", e
	}
	dir, e := os.MkdirTemp(s.c.WorkDir, "request-")
	if e != nil {
		return "", "", problem(http.StatusServiceUnavailable, "storage_unavailable", "Temporary storage unavailable.", nil)
	}
	defer func() {
		finish := metrics.Start(r.Context(), metrics.Cleanup)
		cleanupErr := s.cleanup(dir)
		finish(metrics.Code(cleanupErr))
		if cleanupErr != nil {
			text = ""
			err = problem(http.StatusInternalServerError, "cleanup_failed", "Temporary file cleanup failed.", nil)
		}
	}()
	finishUpload := metrics.Start(r.Context(), metrics.Upload)
	format, err = s.upload(r, dir)
	uploadCode := "ok"
	if err != nil {
		uploadCode = err.Code
	}
	finishUpload(uploadCode)
	if err != nil {
		return "", "", err
	}
	_ = rc.SetReadDeadline(time.Time{})
	if e := s.storage(0); e != nil {
		return "", "", e
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.c.ProcessTimeout)
	defer cancel()
	wav, e := audio.Prepare(ctx, dir, s.c.ReserveBytes)
	if e != nil {
		return "", "", processingError(e)
	}
	metrics.FromContext(ctx).RecordAudio(float64(wav.Samples) / audio.Rate)
	// The native inference call is synchronous. Never release its files or slot
	// on cancellation until the call returns and request-local tensors are freed.
	finishInference := metrics.Start(ctx, metrics.Inference)
	text, e = engine.Transcribe(ctx, wav)
	finishInference(metrics.Code(e))
	closeErr := wav.Close()
	if ctx.Err() != nil {
		e = ctx.Err()
	}
	if e != nil {
		return "", "", processingError(e)
	}
	if closeErr != nil {
		return "", "", problem(http.StatusInternalServerError, "internal_error", "Cannot close audio.", nil)
	}
	return text, format, nil
}
