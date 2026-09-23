package audio

import (
	"fmt"
	"io"

	"github.com/madelynnblue/go-dsp/wav"
)

type Audio struct {
	Audio *wav.Wav
}

func Load(r io.Reader) *Audio {
	audio, err := wav.New(r)

	if err != nil {
		return nil
	}

	return &Audio{
		Audio: audio,
	}
}

// Read возвращает отсчёты в диапазоне [-1, 1) с той же нормировкой, что
// soundfile в Python: int16 / 32768. wav.ReadFloats для этого не подходит:
// он переводит int16 в [0, 1], и обратное 2*v-1 даёт (2v+1)/65535 — чуть
// другой масштаб и сдвиг, из-за которого VAD иногда сдвигает границы окон.
func (a *Audio) Read() ([]float32, error) {
	data, err := a.Audio.ReadSamples(a.Audio.Samples)
	if err != nil {
		return []float32{}, err
	}

	switch data := data.(type) {
	case []int16:
		result := make([]float32, len(data))
		for k, v := range data {
			result[k] = float32(v) / 32768
		}
		return result, nil
	case []uint8:
		// 8-битный PCM беззнаковый, тишина = 128.
		result := make([]float32, len(data))
		for k, v := range data {
			result[k] = (float32(v) - 128) / 128
		}
		return result, nil
	case []float32:
		// IEEE float уже в [-1, 1].
		return data, nil
	default:
		return []float32{}, fmt.Errorf("audio: unsupported sample type %T", data)
	}
}
