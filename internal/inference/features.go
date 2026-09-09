package inference

import (
	_ "embed"
	"encoding/json"
	"math"
)

//go:embed feature_coefficients.json
var coefficientsJSON []byte

// GigaAM v3: periodic Hann, 320-point STFT, hop 160, center=false,
// power=2, HTK mel scale, 64 unnormalised filters, natural log clamped at 1e-9.
// Coefficients preserve the reference PyTorch float32 rounding exactly. The
// deterministic DFT uses windowed float32 samples and double accumulators.
type Features struct {
	cos, sin [161][320]float64
	window   [320]float32
	mel      [64][161]float32
}

func NewFeatures() *Features {
	f := &Features{}
	var coefficients struct {
		Window  []float32 `json:"window"`
		Filters []struct {
			Bin    int     `json:"bin"`
			Mel    int     `json:"mel"`
			Weight float32 `json:"weight"`
		} `json:"filters"`
	}
	if err := json.Unmarshal(coefficientsJSON, &coefficients); err != nil || len(coefficients.Window) != 320 {
		panic("invalid compiled frontend coefficients")
	}
	copy(f.window[:], coefficients.Window)
	for _, v := range coefficients.Filters {
		f.mel[v.Mel][v.Bin] = v.Weight
	}
	for k := 0; k < 161; k++ {
		for n := 0; n < 320; n++ {
			a := 2 * math.Pi * float64(k*n) / 320
			f.cos[k][n] = math.Cos(a)
			f.sin[k][n] = math.Sin(a)
		}
	}
	return f
}
func (f *Features) Compute(x []float32) ([]float32, int) {
	if len(x) < 320 {
		p := make([]float32, 320)
		copy(p, x)
		x = p
	}
	frames := (len(x)-320)/160 + 1
	y := make([]float32, 64*frames)
	for t := 0; t < frames; t++ {
		var windowed [320]float32
		for n := range windowed {
			windowed[n] = x[t*160+n] * f.window[n]
		}
		var power [161]float32
		for k := range power {
			var re, im float64
			for n, v := range windowed {
				re += float64(v) * f.cos[k][n]
				im += float64(v) * f.sin[k][n]
			}
			power[k] = float32(re*re + im*im)
		}
		for m := 0; m < 64; m++ {
			var sum float32
			for k, v := range power {
				sum += v * f.mel[m][k]
			}
			y[m*frames+t] = float32(math.Log(math.Max(1e-9, math.Min(1e9, float64(sum)))))
		}
	}
	return y, frames
}
