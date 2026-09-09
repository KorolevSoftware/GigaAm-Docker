package api

import (
	"context"
	"net/http"
)

func (s *Server) Ready(engine Engine) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.engine = engine
	s.state = ""
}

func (s *Server) Unavailable() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state = "model_unavailable"
}

func (s *Server) availabilityLocked(transcription bool) *Error {
	if transcription && s.stopping {
		return problem(http.StatusServiceUnavailable, "service_stopping", "Service is shutting down.", nil)
	}
	if transcription && len(s.pending) > 0 {
		return problem(http.StatusServiceUnavailable, "cleanup_pending", "Temporary file cleanup is pending.", nil)
	}
	if s.state != "" {
		message := "Model initialization is in progress."
		if s.state == "model_unavailable" {
			message = "Model is unavailable; check service logs."
		}
		return problem(http.StatusServiceUnavailable, s.state, message, nil)
	}
	return nil
}

func (s *Server) acquire() (Engine, *Error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e := s.availabilityLocked(true); e != nil {
		return nil, e
	}
	select {
	case s.slots <- struct{}{}:
		s.active++
		return s.engine, nil
	default:
		return nil, problem(http.StatusTooManyRequests, "service_busy", "All transcription slots are busy.", nil)
	}
}

func (s *Server) release() {
	s.mu.Lock()
	defer s.mu.Unlock()
	<-s.slots
	s.active--
	if s.stopping && s.active == 0 {
		select {
		case <-s.drained:
		default:
			close(s.drained)
		}
	}
}

// Drain stops admission and waits for native calls and request cleanup.
func (s *Server) Drain(ctx context.Context) {
	s.mu.Lock()
	s.stopping = true
	if s.active == 0 {
		select {
		case <-s.drained:
		default:
			close(s.drained)
		}
	}
	s.mu.Unlock()
	select {
	case <-s.drained:
	case <-ctx.Done():
		s.cancel()
		<-s.drained
	}
	s.cancel()
}
