// Package models locates preinstalled model files. Downloads happen at image build time.
package models

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/diagnostics"
)

//go:embed manifest.json
var manifestJSON []byte

type File struct {
	Name   string `json:"name"`
	URL    string `json:"url"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}
type Bundle struct {
	ID           string `json:"id"`
	Revision     string `json:"revision"`
	Precision    string `json:"precision"`
	Architecture string `json:"architecture"`
	Files        []File `json:"files"`
}
type Manifest struct {
	Bundles []Bundle `json:"bundles"`
}

func ManifestData() (Manifest, error) {
	var m Manifest
	e := json.Unmarshal(manifestJSON, &m)
	return m, e
}
func Directory(root string, b Bundle) string {
	return filepath.Join(root, b.ID, b.Revision, b.Precision)
}

// Locate checks that the selected ASR and VAD files are installed, without
// creating directories, downloading files, or reading model contents.
func Locate(root, model, precision string) (string, string, error) {
	manifest, err := ManifestData()
	if err != nil {
		return "", "", diagnostics.Wrap("manifest_invalid", err)
	}
	var asr, vad string
	for _, bundle := range manifest.Bundles {
		isASR := bundle.ID == model && bundle.Precision == precision
		isVAD := bundle.ID == "silero-vad" && bundle.Precision == "fp32"
		if !isASR && !isVAD {
			continue
		}
		dir := Directory(root, bundle)
		for _, file := range bundle.Files {
			info, err := os.Stat(filepath.Join(dir, file.Name))
			if err != nil {
				return "", "", diagnostics.Wrap("model_files_unavailable", err)
			}
			if !info.Mode().IsRegular() {
				return "", "", diagnostics.Wrap("model_files_unavailable", fmt.Errorf("expected regular model file: %s", file.Name))
			}
		}
		if bundle.ID == model {
			asr = dir
		} else {
			vad = dir
		}
	}
	if asr == "" || vad == "" {
		return "", "", diagnostics.Wrap("manifest_invalid", fmt.Errorf("required model bundle missing"))
	}
	return asr, vad, nil
}
