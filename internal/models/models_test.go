package models

import (
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/diagnostics"
)

func TestManifestPinned(t *testing.T) {
	m, e := ManifestData()
	if e != nil {
		t.Fatal(e)
	}
	if len(m.Bundles) != 3 {
		t.Fatal("expected both ASR models and VAD")
	}
	for _, b := range m.Bundles {
		if len(b.Revision) != 40 || b.Precision != "fp32" {
			t.Fatal(b.ID)
		}
		for _, f := range b.Files {
			if !strings.Contains(f.URL, b.Revision) || f.Size < 1 {
				t.Fatal(f.Name)
			}
			if h, e := hex.DecodeString(f.SHA256); e != nil || len(h) != 32 {
				t.Fatal(f.SHA256)
			}
		}
	}
}
func TestLocatePreinstalledModels(t *testing.T) {
	root := filepath.Join(t.TempDir(), "models")
	_, _, err := Locate(root, "gigaam-v3-e2e-rnnt")
	if !errors.Is(err, os.ErrNotExist) || diagnostics.Code(err) != "model_files_unavailable" {
		t.Fatalf("missing model cause lost: %v", err)
	}
	if _, err := os.Stat(root); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("Locate created a model directory")
	}
	manifest, err := ManifestData()
	if err != nil {
		t.Fatal(err)
	}
	for _, bundle := range manifest.Bundles {
		if bundle.ID == "gigaam-v3-e2e-ctc" {
			continue
		}
		dir := Directory(root, bundle)
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		for _, file := range bundle.Files {
			if err := os.WriteFile(filepath.Join(dir, file.Name), nil, 0400); err != nil {
				t.Fatal(err)
			}
		}
	}
	asr, vad, err := Locate(root, "gigaam-v3-e2e-rnnt")
	if err != nil || asr == "" || vad == "" || asr == vad {
		t.Fatalf("invalid model directories: %q %q %v", asr, vad, err)
	}
	if _, _, err := Locate(root, "gigaam-v3-e2e-ctc"); err == nil {
		t.Fatal("missing CTC model accepted")
	}
}
