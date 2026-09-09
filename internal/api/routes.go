package api

import "net/http"

func (s *Server) Handler() http.Handler {
	api := http.NewServeMux()
	api.HandleFunc("GET /v1/models", s.models)
	api.HandleFunc("POST /v1/audio/transcriptions", s.transcribe)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.health)
	mux.Handle("/v1/", s.authorize(api))
	return mux
}
