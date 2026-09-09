// Package diagnostics classifies internal failures without logging private paths,
// URLs, credentials, or native runtime error messages.
package diagnostics

import (
	"context"
	"errors"
	"os"
)

type Error struct {
	Code string
	Err  error
}

func (e *Error) Error() string { return e.Code + ": " + e.Err.Error() }
func (e *Error) Unwrap() error { return e.Err }

func Wrap(code string, err error) error {
	if err == nil {
		return nil
	}
	return &Error{Code: code, Err: err}
}

// Code is safe for logs; the original cause remains available via errors.Is/As.
func Code(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, context.Canceled):
		return "canceled"
	case errors.Is(err, context.DeadlineExceeded):
		return "timeout"
	}
	var diagnostic *Error
	if errors.As(err, &diagnostic) {
		return diagnostic.Code
	}
	switch {
	case errors.Is(err, os.ErrNotExist):
		return "file_missing"
	case errors.Is(err, os.ErrPermission):
		return "permission_denied"
	default:
		return "internal_error"
	}
}
