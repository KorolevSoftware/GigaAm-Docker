package api

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
)

func (s *Server) upload(r *http.Request, dir string) (string, *Error) {
	if r.URL.RawQuery != "" {
		return "", problem(http.StatusBadRequest, "unsupported_parameter", "Query parameters are unsupported.", nil)
	}
	// Bound multipart framing as well as the actual file. MultipartReader streams
	// directly to the dedicated request directory; ParseMultipartForm is not used.
	r.Body = http.MaxBytesReader(nil, r.Body, s.c.MaxFile+(1<<20))
	mr, e := r.MultipartReader()
	if e != nil {
		return "", problem(http.StatusBadRequest, "invalid_multipart", "Expected multipart/form-data.", nil)
	}
	fields := map[string]string{}
	seen := map[string]bool{}
	hasFile := false
	for {
		p, e := mr.NextPart()
		if e == io.EOF {
			break
		}
		if e != nil {
			return "", uploadError(e)
		}
		name := p.FormName()
		if seen[name] {
			return "", problem(http.StatusBadRequest, "duplicate_parameter", "Duplicate multipart field.", nil)
		}
		seen[name] = true
		if name == "file" {
			if p.FileName() == "" {
				return "", problem(http.StatusBadRequest, "invalid_file", "File field must contain a file.", "file")
			}
			hasFile = true
			if err := s.saveUpload(r.Context(), dir, p); err != nil {
				return "", err
			}
		} else {
			switch name {
			case "model", "language", "response_format", "stream":
			default:
				return "", problem(http.StatusBadRequest, "unsupported_parameter", "Unsupported multipart field.", nil)
			}
			if p.FileName() != "" {
				return "", problem(http.StatusBadRequest, "invalid_parameter", "Expected a text field.", name)
			}
			b, e := io.ReadAll(io.LimitReader(p, 4097))
			if e != nil {
				return "", uploadError(e)
			}
			if len(b) > 4096 {
				return "", problem(http.StatusBadRequest, "invalid_parameter", "Parameter is too long.", name)
			}
			fields[name] = string(b)
		}
		if e = p.Close(); e != nil {
			return "", uploadError(e)
		}
	}
	return s.validateFields(fields, hasFile)
}

func (s *Server) saveUpload(ctx context.Context, dir string, source io.Reader) *Error {
	f, e := os.OpenFile(filepath.Join(dir, "source.bin"), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if e != nil {
		return problem(http.StatusServiceUnavailable, "storage_unavailable", "Cannot store upload.", nil)
	}
	n, copyErr := io.Copy(f, io.LimitReader(source, s.c.MaxFile+1))
	metrics.FromContext(ctx).RecordUpload(n)
	closeErr := f.Close()
	if n > s.c.MaxFile {
		return problem(http.StatusRequestEntityTooLarge, "file_too_large", "File exceeds size limit.", "file")
	}
	if copyErr != nil {
		return uploadError(copyErr)
	}
	if closeErr != nil {
		return problem(http.StatusServiceUnavailable, "storage_unavailable", "Cannot store upload.", nil)
	}
	if n == 0 {
		return problem(http.StatusBadRequest, "empty_file", "File is empty.", "file")
	}
	return nil
}

func (s *Server) validateFields(fields map[string]string, hasFile bool) (string, *Error) {
	if !hasFile {
		return "", problem(http.StatusBadRequest, "missing_file", "file is required.", "file")
	}
	if fields["model"] == "" {
		return "", problem(http.StatusBadRequest, "missing_model", "model is required.", "model")
	}
	if fields["model"] != s.c.Model {
		return "", problem(http.StatusBadRequest, "model_mismatch", "model must match the server model.", "model")
	}
	if v, ok := fields["language"]; ok && v != "ru" {
		return "", problem(http.StatusBadRequest, "unsupported_language", "Only ru is supported.", "language")
	}
	if v, ok := fields["stream"]; ok && v != "false" {
		return "", problem(http.StatusBadRequest, "unsupported_stream", "Only stream=false is supported.", "stream")
	}
	format := "json"
	if v, ok := fields["response_format"]; ok {
		format = v
	}
	if format != "json" && format != "text" {
		return "", problem(http.StatusBadRequest, "unsupported_response_format", "Use json or text.", "response_format")
	}
	return format, nil
}
