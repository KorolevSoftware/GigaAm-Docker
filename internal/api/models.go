package api

import "net/http"

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	err := s.availabilityLocked(false)
	s.mu.Unlock()
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{{
			"id":       s.c.Model,
			"object":   "model",
			"created":  0,
			"owned_by": "local",
		}},
	})
}
