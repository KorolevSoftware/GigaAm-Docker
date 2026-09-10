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
	expected := map[string]bool{
		"gigaam-v3-e2e-rnnt/fp32": false, "gigaam-v3-e2e-rnnt/int8": false,
		"gigaam-v3-e2e-ctc/fp32": false, "gigaam-v3-e2e-ctc/int8": false,
		"silero-vad/fp32": false,
	}
	if len(m.Bundles) != len(expected) {
		t.Fatal("expected FP32/INT8 ASR models and FP32 VAD")
	}
	for _, b := range m.Bundles {
		key := b.ID + "/" + b.Precision
		seen, ok := expected[key]
		if !ok || seen {
			t.Fatalf("unexpected or duplicate bundle: %s", key)
		}
		expected[key] = true
		if len(b.Revision) != 40 {
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
	_, _, err := Locate(root, "gigaam-v3-e2e-rnnt", "fp32")
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
		if bundle.ID == "gigaam-v3-e2e-ctc" || bundle.Precision != "fp32" {
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
	asr, vad, err := Locate(root, "gigaam-v3-e2e-rnnt", "fp32")
	if err != nil || asr == "" || vad == "" || asr == vad {
		t.Fatalf("invalid model directories: %q %q %v", asr, vad, err)
	}
	if _, _, err := Locate(root, "gigaam-v3-e2e-ctc", "fp32"); err == nil {
		t.Fatal("missing CTC model accepted")
	}
}

// Selecting INT8 must never silently load installed FP32 weights, or vice versa.
func TestLocatePrecisionIsolation(t *testing.T) {
	manifest, err := ManifestData()
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range []string{"gigaam-v3-e2e-rnnt", "gigaam-v3-e2e-ctc"} {
		for _, installed := range []string{"fp32", "int8"} {
			t.Run(model+"/"+installed, func(t *testing.T) {
				root := t.TempDir()
				var expectedASR, expectedVAD string
				for _, b := range manifest.Bundles {
					if !(b.ID == model && b.Precision == installed) && b.ID != "silero-vad" {
						continue
					}
					dir := Directory(root, b)
					if err := os.MkdirAll(dir, 0700); err != nil {
						t.Fatal(err)
					}
					for _, f := range b.Files {
						if err := os.WriteFile(filepath.Join(dir, f.Name), nil, 0400); err != nil {
							t.Fatal(err)
						}
					}
					if b.ID == model {
						expectedASR = dir
					} else {
						expectedVAD = dir
					}
				}
				asr, vad, err := Locate(root, model, installed)
				if err != nil || asr != expectedASR || vad != expectedVAD {
					t.Fatalf("asr=%q vad=%q err=%v", asr, vad, err)
				}
				other := "int8"
				if installed == "int8" {
					other = "fp32"
				}
				if _, _, err = Locate(root, model, other); !errors.Is(err, os.ErrNotExist) || diagnostics.Code(err) != "model_files_unavailable" {
					t.Fatalf("missing precision fell back to installed weights: %v", err)
				}
			})
		}
	}
}
