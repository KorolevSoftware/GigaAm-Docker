package api

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"os"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
)

// 499 is a non-standard status and has no net/http constant.
const statusClientClosedRequest = 499

type Error struct {
	Status  int    `json:"-"`
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   any    `json:"param"`
	Code    string `json:"code"`
}

func problem(status int, code, message string, param any) *Error {
	t := "server_error"
	if status >= http.StatusBadRequest && status < http.StatusInternalServerError {
		t = "invalid_request_error"
	}
	return &Error{status, message, t, param, code}
}

func (e *Error) Error() string { return e.Code }

func uploadError(e error) *Error {
	var tooLarge *http.MaxBytesError
	if errors.As(e, &tooLarge) {
		return problem(http.StatusRequestEntityTooLarge, "file_too_large", "Request exceeds size limit.", "file")
	}
	var ne net.Error
	if errors.As(e, &ne) && ne.Timeout() {
		return problem(http.StatusRequestTimeout, "upload_timeout", "Upload timed out.", "file")
	}
	var pe *os.PathError
	if errors.As(e, &pe) {
		return problem(http.StatusServiceUnavailable, "storage_unavailable", "Cannot store upload.", nil)
	}
	return problem(http.StatusBadRequest, "invalid_upload", "Incomplete or malformed upload.", "file")
}

func processingError(e error) *Error {
	switch {
	case errors.Is(e, context.DeadlineExceeded):
		return problem(http.StatusGatewayTimeout, "transcription_timeout", "Transcription timed out.", nil)
	case errors.Is(e, context.Canceled):
		return problem(statusClientClosedRequest, "request_cancelled", "Request cancelled.", nil)
	case errors.Is(e, audio.ErrDuration):
		return problem(http.StatusUnprocessableEntity, "duration_exceeded", "Recording exceeds duration limit.", "file")
	case errors.Is(e, audio.ErrNoAudio):
		return problem(http.StatusUnprocessableEntity, "no_audio_stream", "File contains no audio stream.", "file")
	case errors.Is(e, audio.ErrUnavailable):
		return problem(http.StatusServiceUnavailable, "media_unavailable", "FFmpeg or FFprobe is unavailable.", nil)
	case errors.Is(e, audio.ErrStorage):
		return problem(http.StatusServiceUnavailable, "storage_unavailable", "Temporary storage is full.", nil)
	case errors.Is(e, audio.ErrInvalid):
		return problem(http.StatusUnprocessableEntity, "invalid_audio", "Unsupported or damaged audio.", "file")
	default:
		return problem(http.StatusInternalServerError, "transcription_failed", "Transcription failed.", nil)
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, e *Error) { writeJSON(w, e.Status, map[string]any{"error": e}) }
