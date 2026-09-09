package api

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (s *Server) CleanStartup() error {
	if e := os.MkdirAll(s.c.WorkDir, 0700); e != nil {
		return e
	}
	entries, e := os.ReadDir(s.c.WorkDir)
	if e != nil {
		return e
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), "request-") {
			_ = s.cleanup(filepath.Join(s.c.WorkDir, entry.Name()))
		}
	}
	return nil
}

func (s *Server) RetryCleanup(ctx context.Context) {
	t := time.NewTicker(time.Second)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			s.mu.Lock()
			paths := make([]string, 0, len(s.pending))
			for p := range s.pending {
				paths = append(paths, p)
			}
			s.mu.Unlock()
			for _, p := range paths {
				if s.removeAll(p) == nil {
					s.mu.Lock()
					delete(s.pending, p)
					s.mu.Unlock()
				}
			}
		}
	}
}

func (s *Server) cleanup(p string) error {
	e := s.removeAll(p)
	if e != nil {
		s.mu.Lock()
		s.pending[p] = true
		s.mu.Unlock()
	}
	return e
}

func (s *Server) storage(required int64) *Error {
	n, e := s.freeBytes(s.c.WorkDir)
	if e != nil || n < uint64(required+s.c.ReserveBytes) {
		return problem(http.StatusServiceUnavailable, "storage_unavailable", "Insufficient temporary disk space.", nil)
	}
	return nil
}
