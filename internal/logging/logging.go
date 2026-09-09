// Package logging configures the service's JSON log schema.
package logging

import (
	"io"
	"os"
	"strings"
	"time"

	"github.com/rs/zerolog"
)

// Configure zerolog's process-wide field names once, before any logging starts.
func init() {
	zerolog.MessageFieldName = "msg"
	zerolog.TimeFieldFormat = time.RFC3339Nano
	zerolog.LevelFieldMarshalFunc = func(level zerolog.Level) string {
		return strings.ToUpper(level.String())
	}
}

var Log = New(os.Stdout)

// New shares a synchronized writer with every derived request logger.
func New(out io.Writer) zerolog.Logger {
	return zerolog.New(zerolog.SyncWriter(out)).Level(zerolog.InfoLevel).With().Timestamp().Logger()
}
