package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	Model           string
	ModelDir        string
	WorkDir         string
	Key             string
	Port            string
	ORTLibrary      string
	Concurrency     int
	Threads         int
	MaxFile         int64
	ReserveBytes    int64
	MaxDuration     time.Duration
	UploadTimeout   time.Duration
	ProcessTimeout  time.Duration
	ResponseTimeout time.Duration
	GracePeriod     time.Duration
	Chunk           time.Duration
	Overlap         time.Duration
	Padding         time.Duration
	MinSilence      time.Duration
	VADThreshold    float64
}

func defaults() Config {
	return Config{
		Model:           "gigaam-v3-e2e-rnnt",
		ModelDir:        "/models",
		WorkDir:         "/work",
		Port:            "8080",
		ORTLibrary:      "/usr/local/lib/libonnxruntime.so",
		Concurrency:     1,
		Threads:         4,
		MaxFile:         20 << 30,
		ReserveBytes:    512 << 20,
		MaxDuration:     4 * time.Hour,
		UploadTimeout:   30 * time.Minute,
		ProcessTimeout:  time.Hour,
		ResponseTimeout: 5 * time.Minute,
		GracePeriod:     90 * time.Second,
		Chunk:           20 * time.Second,
		Overlap:         time.Second,
		Padding:         300 * time.Millisecond,
		MinSilence:      500 * time.Millisecond,
		VADThreshold:    0.5,
	}
}

func Load() (Config, error) {
	c := defaults()
	for k, p := range map[string]*string{
		"GIGAAM_MODEL":       &c.Model,
		"GIGAAM_MODEL_DIR":   &c.ModelDir,
		"GIGAAM_WORK_DIR":    &c.WorkDir,
		"GIGAAM_API_KEY":     &c.Key,
		"GIGAAM_PORT":        &c.Port,
		"GIGAAM_ORT_LIBRARY": &c.ORTLibrary,
	} {
		if v, ok := os.LookupEnv(k); ok {
			*p = v
		}
	}
	if p := os.Getenv("GIGAAM_API_KEY_FILE"); p != "" {
		b, e := os.ReadFile(p)
		if e != nil {
			return c, fmt.Errorf("cannot read GIGAAM_API_KEY_FILE")
		}
		c.Key = strings.TrimSpace(string(b))
	}
	if c.Key == "" {
		return c, fmt.Errorf("GIGAAM_API_KEY or GIGAAM_API_KEY_FILE is required")
	}
	if c.Model != "gigaam-v3-e2e-rnnt" && c.Model != "gigaam-v3-e2e-ctc" {
		return c, fmt.Errorf("GIGAAM_MODEL must be gigaam-v3-e2e-rnnt or gigaam-v3-e2e-ctc")
	}
	if n, e := strconv.Atoi(c.Port); e != nil || n < 1 || n > 65535 {
		return c, fmt.Errorf("invalid GIGAAM_PORT")
	}
	for k, p := range map[string]*int{
		"GIGAAM_CONCURRENCY":  &c.Concurrency,
		"GIGAAM_ONNX_THREADS": &c.Threads,
	} {
		if v, ok := os.LookupEnv(k); ok {
			n, e := strconv.Atoi(v)
			if e != nil || n < 1 {
				return c, fmt.Errorf("invalid %s", k)
			}
			*p = n
		}
	}
	if c.Concurrency > 64 || c.Threads > 256 {
		return c, fmt.Errorf("concurrency <= 64, threads <= 256 required")
	}
	for k, p := range map[string]*int64{
		"GIGAAM_MAX_FILE_BYTES": &c.MaxFile,
		"GIGAAM_RESERVE_BYTES":  &c.ReserveBytes,
	} {
		if v, ok := os.LookupEnv(k); ok {
			n, e := strconv.ParseInt(v, 10, 64)
			if e != nil || n < 1 || n > 1<<40 {
				return c, fmt.Errorf("invalid %s", k)
			}
			*p = n
		}
	}
	for k, p := range map[string]*time.Duration{
		"GIGAAM_MAX_DURATION":     &c.MaxDuration,
		"GIGAAM_UPLOAD_TIMEOUT":   &c.UploadTimeout,
		"GIGAAM_PROCESS_TIMEOUT":  &c.ProcessTimeout,
		"GIGAAM_RESPONSE_TIMEOUT": &c.ResponseTimeout,
		"GIGAAM_GRACE_PERIOD":     &c.GracePeriod,
		"GIGAAM_CHUNK_DURATION":   &c.Chunk,
		"GIGAAM_OVERLAP":          &c.Overlap,
		"GIGAAM_PADDING":          &c.Padding,
		"GIGAAM_MIN_SILENCE":      &c.MinSilence,
	} {
		if v, ok := os.LookupEnv(k); ok {
			n, e := time.ParseDuration(v)
			if e != nil || n <= 0 {
				return c, fmt.Errorf("invalid %s", k)
			}
			*p = n
		}
	}
	if c.Chunk > 30*time.Second || c.Chunk < time.Second || c.Overlap >= c.Chunk/2 || c.Padding >= c.Chunk/2 || c.MinSilence >= c.Chunk || c.MaxDuration > 24*time.Hour {
		return c, fmt.Errorf("invalid audio window/duration limits (chunk 1s..30s)")
	}
	if v, ok := os.LookupEnv("GIGAAM_VAD_THRESHOLD"); ok {
		n, e := strconv.ParseFloat(v, 64)
		if e != nil || !(n > 0 && n < 1) {
			return c, fmt.Errorf("invalid GIGAAM_VAD_THRESHOLD")
		}
		c.VADThreshold = n
	}
	var e error
	c.WorkDir, e = canonicalPath(c.WorkDir)
	if e != nil {
		return c, e
	}
	c.ModelDir, e = canonicalPath(c.ModelDir)
	if e != nil {
		return c, e
	}
	if c.WorkDir == "/" || c.ModelDir == "/" || within(c.WorkDir, c.ModelDir) || within(c.ModelDir, c.WorkDir) {
		return c, fmt.Errorf("work and model directories must be separate, non-root directories")
	}
	return c, nil
}
func within(a, b string) bool { return a == b || strings.HasPrefix(b, a+string(os.PathSeparator)) }

// Resolve existing symlink ancestors even when the final directory is not yet
// created, so aliases cannot put request audio inside the persistent cache.
func canonicalPath(path string) (string, error) {
	abs, e := filepath.Abs(path)
	if e != nil {
		return "", e
	}
	current := abs
	var suffix []string
	for {
		resolved, err := filepath.EvalSymlinks(current)
		if err == nil {
			for i := len(suffix) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, suffix[i])
			}
			return resolved, nil
		}
		if os.IsPermission(err) {
			return abs, nil
		} // readiness handles inaccessible storage after HTTP starts
		if !os.IsNotExist(err) {
			return "", fmt.Errorf("cannot resolve storage directory")
		}
		parent := filepath.Dir(current)
		if parent == current {
			return "", err
		}
		suffix = append(suffix, filepath.Base(current))
		current = parent
	}
}
