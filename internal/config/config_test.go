package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultsAndValidation(t *testing.T) {
	t.Setenv("GIGAAM_API_KEY", "test-key")
	c, e := Load()
	if e != nil {
		t.Fatal(e)
	}
	if c.Model != "gigaam-v3-e2e-rnnt" || c.MaxFile != 20<<30 {
		t.Fatal(c.Model, c.MaxFile)
	}
	for _, tt := range [][2]string{{"GIGAAM_MODEL", "unknown"}, {"GIGAAM_PORT", "65536"}, {"GIGAAM_CONCURRENCY", "0"}, {"GIGAAM_CHUNK_DURATION", "31s"}, {"GIGAAM_OVERLAP", "20s"}, {"GIGAAM_VAD_THRESHOLD", "NaN"}, {"GIGAAM_WORK_DIR", "/models/work"}} {
		t.Run(tt[0], func(t *testing.T) {
			t.Setenv(tt[0], tt[1])
			if _, e := Load(); e == nil {
				t.Fatal("invalid setting accepted")
			}
		})
	}
}
func TestKeyFile(t *testing.T) {
	t.Setenv("GIGAAM_API_KEY_FILE", "/does-not-exist")
	if _, e := Load(); e == nil {
		t.Fatal("missing secret ignored")
	}
}

func TestStorageSymlinkAliases(t *testing.T) {
	t.Setenv("GIGAAM_API_KEY", "key")
	dir := t.TempDir()
	cache := filepath.Join(dir, "models")
	if e := os.Mkdir(cache, 0700); e != nil {
		t.Fatal(e)
	}
	alias := filepath.Join(dir, "alias")
	if e := os.Symlink(cache, alias); e != nil {
		t.Fatal(e)
	}
	t.Setenv("GIGAAM_MODEL_DIR", cache)
	t.Setenv("GIGAAM_WORK_DIR", filepath.Join(alias, "requests"))
	if _, e := Load(); e == nil {
		t.Fatal("aliased storage overlap accepted")
	}
}
