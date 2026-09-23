package features

import (
	"encoding/json"
	"fmt"
	"os"
)

// FilterEntry — один ненулевой коэффициент мел-матрицы.
type FilterEntry struct {
	Bin    int     `json:"bin"`    // индекс частотного бина rfft, 0..NFFT/2
	Mel    int     `json:"mel"`    // индекс мел-полосы, 0..63
	Weight float32 `json:"weight"` // вес бина в этой полосе
}

// Coefficients — содержимое feature_coefficients.json.
type Coefficients struct {
	Torch      string        `json:"torch"`
	Torchaudio string        `json:"torchaudio"`
	SampleRate int           `json:"sample_rate"`
	NFFT       int           `json:"n_fft"`
	HopLength  int           `json:"hop_length"`
	Center     bool          `json:"center"`
	Window     []float32     `json:"window"`
	Filters    []FilterEntry `json:"filters"`
}

const NMels = 64

// Load читает JSON и проверяет, что он совпадает с тем, что ожидает фронтенд.
func Load(path string) (*Coefficients, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Coefficients
	if err := json.Unmarshal(data, &c); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(c.Window) != c.NFFT {
		return nil, fmt.Errorf("window length %d != n_fft %d", len(c.Window), c.NFFT)
	}
	bins := c.NFFT/2 + 1
	for i, f := range c.Filters {
		if f.Bin < 0 || f.Bin >= bins || f.Mel < 0 || f.Mel >= NMels {
			return nil, fmt.Errorf("filters[%d] out of range: bin=%d mel=%d", i, f.Bin, f.Mel)
		}
	}
	return &c, nil
}

// DenseFilters строит плотную матрицу [NFFT/2+1][NMels], как в Python:
// filters = np.zeros((161, 64)); filters[bin, mel] = weight
func (c *Coefficients) DenseFilters() [][]float32 {
	bins := c.NFFT/2 + 1
	m := make([][]float32, bins)
	for i := range m {
		m[i] = make([]float32, NMels)
	}
	for _, f := range c.Filters {
		m[f.Bin][f.Mel] = f.Weight
	}
	return m
}
