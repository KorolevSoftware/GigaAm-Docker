package api

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"net/http"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
)

func (s *Server) authorize(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Reject unread bodies immediately. Otherwise net/http can drain a slow body
		// before sending 401/429/503, defeating admission control and upload deadlines.
		if r.Body != nil && r.Body != http.NoBody {
			w.Header().Set("Connection", "close")
		}
		var id [16]byte
		if _, e := rand.Read(id[:]); e != nil {
			writeError(w, problem(http.StatusInternalServerError, "internal_error", "Cannot create request ID.", nil))
			return
		}
		rid := hex.EncodeToString(id[:])
		w.Header().Set("X-Request-ID", rid)
		w.Header().Set("Cache-Control", "no-store")
		got := sha256.Sum256([]byte(r.Header.Get("Authorization")))
		want := sha256.Sum256([]byte("Bearer " + s.c.Key))
		if subtle.ConstantTimeCompare(got[:], want[:]) != 1 {
			w.Header().Set("WWW-Authenticate", "Bearer")
			writeError(w, problem(http.StatusUnauthorized, "invalid_api_key", "Invalid bearer API key.", nil))
			return
		}
		ctx := metrics.NewContext(r.Context(), rid)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
