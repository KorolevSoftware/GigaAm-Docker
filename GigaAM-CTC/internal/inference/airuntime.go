package airuntime

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/KorolevSoftware/GigaAm-Docker/GigaAM-CTC/internal/features"
	"github.com/microsoft/onnxruntime/go/onnxruntime"
	"gonum.org/v1/gonum/dsp/fourier"
)

const RATE = 16000

type AIRuntimeCTC struct {
	session      *onnxruntime.Session
	outputName   string // первый выход модели: log_probs [1, T/4, словарь]
	coefficients *features.Coefficients
	vocab        []string
	blank        int // в этом экспорте последний токен означает «ничего не выдать»
}

// Segment — результат распознавания одного окна VAD.
type Segment struct {
	Start    int    // начало окна в отсчётах
	Stop     int    // конец окна в отсчётах
	Frames   int    // число кадров признаков (valid)
	Text     string // текст всего окна
	Addition string // только добавленная часть, без дубля перекрытия
}

type AIRuntimeVad struct {
	session *onnxruntime.Session
}

type VadWindow struct {
	Start int
	Stop  int
}

func MakeWindows(probably []float32, audio []float32, chunkSeconds float32) []VadWindow {
	resultWindows := make([]VadWindow, 0, 50)
	chunk := int(chunkSeconds * RATE)
	// Перекрытие 1 с; запас до/после речи 0,3 с; пауза для закрытия окна 0,5 с.
	overlap, pad, silence := RATE, 4800, 8000
	// -1 означает: речевое окно / отсчёт паузы ещё не начаты.
	start, lastVoice, silenceAt := -1, 0, -1
	for frame, prob := range probably {
		// Один результат Silero соответствует 512 новым отсчётам, то есть 32 мс.
		pos := frame * 512
		end := min(pos+512, len(audio))

		// Разные пороги начала речи и тишины уменьшают переключения на границе.
		if prob >= 0.5 {
			if start < 0 {
				start = max(0, pos-pad)
			}
			lastVoice = end
			silenceAt = -1
		} else if prob < 0.35 && start >= 0 && silenceAt < 0 {
			silenceAt = pos
		}

		if start < 0 {
			continue
		}

		if silenceAt >= 0 && end-silenceAt >= silence {
			// Пауза достаточно длинная: закрываем окно, сохраняя запас после речи.
			if stop := min(lastVoice+pad, end); stop > start {
				resultWindows = append(resultWindows, VadWindow{Start: start, Stop: stop})
			}
			start = -1
			silenceAt = -1
			continue
		}

		if end-start >= chunk {
			// Речь непрерывная, но лимит длины достигнут. Режем у тихой точки.
			stop := quietBoundary(audio, start, start+chunk)
			resultWindows = append(resultWindows, VadWindow{Start: start, Stop: stop})
			// Следующее окно повторит последнюю секунду предыдущего.
			start = stop - overlap
		}
	}

	// Конец файла: выдаём оставшуюся речь, даже если после неё не было паузы.
	if start >= 0 && len(audio) > start {
		resultWindows = append(resultWindows, VadWindow{Start: start, Stop: len(audio)})
	}

	return resultWindows
}

func quietBoundary(audio []float32, start, stop int) int {
	// Проверяем последние 3 секунды, но не раньше середины текущего окна.
	from := max(start+(stop-start)/2, stop-3*RATE)
	x := audio[from:stop]
	const width = 1600 // 100ms power window
	if len(x) < width {
		return stop
	}
	// Префиксные суммы энергии, как np.cumsum в Python: cs[k] = Σ x[:k]².
	// Энергия участка [i, i+width) = cs[i+width]-cs[i]. Скользящая сумма
	// (+new²-old²) накапливает другую ошибку округления и на тихих участках
	// может сдвинуть минимум на соседний отсчёт.
	cs := make([]float64, len(x)+1)
	for i, v := range x {
		cs[i+1] = cs[i] + float64(v)*float64(v)
	}
	total := cs[len(x)]
	if total == 0 {
		return stop
	}

	// При равной энергии выбираем последний минимум (<=).
	best, bestAt := math.Inf(1), 0
	for i := 0; i+width <= len(x); i++ {
		if power := cs[i+width] - cs[i]; power <= best {
			best, bestAt = power, i
		}
	}

	if best/width < total/float64(len(x))*0.03 {
		return from + bestAt + width/2
	}
	return stop
}

func (vad *AIRuntimeVad) Run(ctx context.Context, audio []float32) ([]float32, error) {
	// Тензоры ссылаются на эти Go-слайсы без копирования, поэтому создаём их
	// один раз и между вызовами меняем данные на месте.
	//  https://github.com/snakers4/silero-vad/blob/60b7ffa243625ebdc1070275a29f18c87843786a/examples/onnx_sequence/run.py#L20
	rawData := make([]float32, 576)
	stateData := make([]float32, 2*128)

	data, err := onnxruntime.CreateTensor([]int64{1, 576}, rawData)
	if err != nil {
		return nil, fmt.Errorf("vad input tensor: %w", err)
	}
	defer data.Close()

	state, err := onnxruntime.CreateTensor([]int64{2, 1, 128}, stateData)
	if err != nil {
		return nil, fmt.Errorf("vad state tensor: %w", err)
	}
	defer state.Close()

	sr, err := onnxruntime.CreateTensor([]int64{1}, []int64{RATE})
	if err != nil {
		return nil, fmt.Errorf("vad sr tensor: %w", err)
	}
	defer sr.Close()

	inputs := map[string]*onnxruntime.Tensor{"input": data, "state": state, "sr": sr}
	outputNames := []string{"output", "stateN"}

	result := make([]float32, 0, len(audio)/512+1)
	for i := 0; i < len(audio); i += 512 {
		// Вход: 64 прошлых + 512 новых отсчётов. Хвост файла дополняем нулями.
		copy(rawData[:64], rawData[512:])
		n := copy(rawData[64:], audio[i:])
		clear(rawData[64+n:])

		output, err := vad.session.Run(ctx, inputs, outputNames)
		if err != nil {
			return nil, fmt.Errorf("vad run: %w", err)
		}

		prob, err := readVadOutput(output, stateData)
		output["output"].Close()
		output["stateN"].Close()
		if err != nil {
			return nil, err
		}

		result = append(result, prob)
	}

	return result, nil
}

// readVadOutput достаёт вероятность речи и копирует новое состояние в stateData,
// на который ссылается входной тензор state.
func readVadOutput(output map[string]*onnxruntime.Tensor, stateData []float32) (float32, error) {
	stateN, err := onnxruntime.TensorData[float32](output["stateN"])
	if err != nil {
		return 0, fmt.Errorf("vad stateN: %w", err)
	}
	if len(stateN) != len(stateData) {
		return 0, fmt.Errorf("vad stateN size %d, want %d", len(stateN), len(stateData))
	}
	copy(stateData, stateN)

	out, err := onnxruntime.TensorData[float32](output["output"])
	if err != nil {
		return 0, fmt.Errorf("vad output: %w", err)
	}
	if len(out) == 0 {
		return 0, fmt.Errorf("vad output is empty")
	}
	prob := out[0] // Вероятность речи; это ещё не распознавание слов.
	if math.IsNaN(float64(prob)) || math.IsInf(float64(prob), 0) {
		return 0, fmt.Errorf("vad output is not finite: %v", prob)
	}
	return prob, nil
}

func MakeVadSession(vadModelPath string) (*AIRuntimeVad, error) {
	options, err := onnxruntime.NewSessionOptions()

	if err != nil {
		return nil, err
	}

	options.SetIntraOpNumThreads(1)
	options.SetInterOpNumThreads(1)

	session, err := onnxruntime.NewSession(vadModelPath, options)
	if err != nil {
		return nil, err
	}

	vad := &AIRuntimeVad{session: session}

	return vad, nil
}

// MakeCTCSession загружает CTC-модель. provider: "cpu" (по умолчанию), "webgpu"
// (на macOS идёт через Metal) или "coreml". Узлы, которые провайдер не умеет,
// ORT штатно исполняет на CPU.
func MakeCTCSession(path, featureCoefficients, vocabPath, provider string) (*AIRuntimeCTC, error) {
	options, err := onnxruntime.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	// Это потоки для CPU-части ASR.
	options.SetIntraOpNumThreads(2)
	options.SetInterOpNumThreads(1)
	// Подаём по одному окну: фиксируем batch_size=1, чтобы ORT заранее вывел
	// формы тензоров и лучше оптимизировал граф. Длина по времени остаётся динамической.
	options.AddFreeDimensionOverrideByName("batch_size", 1)

	switch provider {
	case "", "cpu":
	case "webgpu":
		if err := options.AppendExecutionProvider("WebGPU", nil); err != nil {
			return nil, err
		}
	case "coreml":
		// Без RequireStaticInputShapes MPSGraph падает при компиляции ('mps.matmul'
		// contracting dimensions differ 1 & 768) и роняет процесс. Со статическими
		// формами CoreML берёт только 349 из 1601 узла, остальное идёт на CPU,
		// поэтому по скорости это почти CPU. Быстрее всего на Mac — "webgpu".
		if err := options.AppendExecutionProvider("CoreML", map[string]string{
			"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU", "RequireStaticInputShapes": "1",
		}); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown ctc provider %q (want cpu, webgpu or coreml)", provider)
	}

	session, err := onnxruntime.NewSession(path, options)
	if err != nil {
		return nil, err
	}

	coef, err := features.Load(featureCoefficients)
	if err != nil {
		session.Close()
		return nil, err
	}

	vocab, err := loadVocab(vocabPath)
	if err != nil {
		session.Close()
		return nil, err
	}

	outputs := session.Outputs()
	if len(outputs) == 0 {
		session.Close()
		return nil, fmt.Errorf("ctc model has no outputs")
	}

	return &AIRuntimeCTC{
		session:      session,
		outputName:   outputs[0].Name,
		coefficients: coef,
		vocab:        vocab,
		blank:        len(vocab) - 1,
	}, nil
}

// loadVocab читает словарь: в каждой строке «токен id», берём всё до последнего пробела.
func loadVocab(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	vocab := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSuffix(scanner.Text(), "\r")
		if i := strings.LastIndex(line, " "); i >= 0 {
			line = line[:i]
		}
		vocab = append(vocab, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read vocab %s: %w", path, err)
	}
	if len(vocab) == 0 {
		return nil, fmt.Errorf("vocab %s is empty", path)
	}
	return vocab, nil
}

// logMel считает признаки [1, 64, T] одним плоским слайсом: out[m*valid+t].
func (ctc *AIRuntimeCTC) logMel(chunk []float32) ([]float32, int) {
	c := ctc.coefficients
	win, hop := c.NFFT, c.HopLength // 320, 160

	valid := 1
	if len(chunk) >= win {
		valid = (len(chunk)-win)/hop + 1
	}
	if len(chunk) < win {
		padded := make([]float32, win)
		copy(padded, chunk)
		chunk = padded
	}

	// FFT для действительного сигнала: сразу отдаёт 161 бин и не выделяет память
	// на кадр. Объект не потокобезопасен, поэтому создаём его на каждый вызов —
	// это дёшево по сравнению с расчётом сотен кадров.
	fft := fourier.NewFFT(win)
	out := make([]float32, features.NMels*valid)
	frame := make([]float64, win) // отдельный буфер: audio не трогаем
	spectrum := make([]complex128, win/2+1)
	power := make([]float64, win/2+1)
	mel := make([]float64, features.NMels)

	for t := 0; t < valid; t++ {
		src := chunk[t*hop : t*hop+win]
		for i := range frame {
			frame[i] = float64(src[i] * c.Window[i])
		}
		fft.Coefficients(spectrum, frame)
		for k, z := range spectrum {
			re, im := real(z), imag(z)
			power[k] = re*re + im*im
		}
		clear(mel)
		for _, e := range c.Filters {
			mel[e.Mel] += power[e.Bin] * float64(e.Weight)
		}
		for m, v := range mel {
			v = min(max(v, 1e-9), 1e9)
			out[m*valid+t] = float32(math.Log(v))
		}
	}
	return out, valid
}

// Run распознаёт окна VAD по порядку и склеивает их текст.
func (ctc *AIRuntimeCTC) Run(ctx context.Context, vadWindows []VadWindow, audio []float32) (string, []Segment, error) {
	merger := NewMerger()
	segments := make([]Segment, 0, len(vadWindows))
	for _, w := range vadWindows {
		text, positions, frames, err := ctc.Recognize(ctx, audio[w.Start:w.Stop])
		if err != nil {
			return "", segments, fmt.Errorf("ctc window %d-%d: %w", w.Start, w.Stop, err)
		}
		addition := merger.Add(w.Start, w.Stop, text, positions)
		segments = append(segments, Segment{
			Start: w.Start, Stop: w.Stop, Frames: frames, Text: text, Addition: addition,
		})
	}
	return merger.Text(), segments, nil
}

// Recognize распознаёт один кусок аудио. positions — примерные начала слов
// в отсчётах относительно начала куска; frames — число кадров признаков.
func (ctc *AIRuntimeCTC) Recognize(ctx context.Context, chunk []float32) (string, []int, int, error) {
	feats, valid := ctc.logMel(chunk)

	featuresT, err := onnxruntime.CreateTensor([]int64{1, features.NMels, int64(valid)}, feats)
	if err != nil {
		return "", nil, 0, fmt.Errorf("features tensor: %w", err)
	}
	defer featuresT.Close()

	lengthsT, err := onnxruntime.CreateTensor([]int64{1}, []int64{int64(valid)})
	if err != nil {
		return "", nil, 0, fmt.Errorf("feature_lengths tensor: %w", err)
	}
	defer lengthsT.Close()

	// Один синхронный вызов CTC на всё окно.
	output, err := ctc.session.Run(ctx, map[string]*onnxruntime.Tensor{
		"features": featuresT, "feature_lengths": lengthsT,
	}, []string{ctc.outputName})
	if err != nil {
		return "", nil, 0, fmt.Errorf("ctc run: %w", err)
	}
	out := output[ctc.outputName]
	defer out.Close()

	logProbs, err := onnxruntime.TensorData[float32](out)
	if err != nil {
		return "", nil, 0, fmt.Errorf("ctc output: %w", err)
	}
	shape := out.Shape()
	if len(shape) != 3 || shape[0] != 1 {
		return "", nil, 0, fmt.Errorf("ctc output shape %v, want [1, T, vocab]", shape)
	}
	if int(shape[2]) != len(ctc.vocab) {
		return "", nil, 0, fmt.Errorf("ctc output size %d and vocabulary size %d differ", shape[2], len(ctc.vocab))
	}
	for _, v := range logProbs {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return "", nil, 0, fmt.Errorf("model output contains NaN or Inf")
		}
	}

	// Времени на выходе примерно в 4 раза меньше, чем кадров признаков.
	frames := min((valid+3)/4, int(shape[1]))
	emissions := greedyCTC(logProbs, frames, len(ctc.vocab), ctc.blank)
	text, positions := textAndPositions(emissions, ctc.vocab)
	return text, positions, valid, nil
}

// greedyCTC выбирает самый вероятный токен каждого кадра, схлопывает одинаковые
// СОСЕДНИЕ токены исходного пути и убирает blank: а,а,blank,а → аа.
// Softmax не нужен: для выбора максимума достаточно log_probs.
func greedyCTC(logProbs []float32, frames, vocabSize, blank int) []Emission {
	emissions := []Emission{}
	prev := -1
	for i := range frames {
		row := logProbs[i*vocabSize : (i+1)*vocabSize]
		best := 0 // как np.argmax: при равенстве берём первый максимум
		for k, v := range row {
			if v > row[best] {
				best = k
			}
		}
		if best != blank && best != prev {
			// Шаг выхода 4*160 = 640 отсчётов = 40 мс; позиция приблизительная.
			emissions = append(emissions, Emission{Token: best, Sample: i * 640})
		}
		prev = best
	}
	return emissions
}

func Load(ctcModelPath, vadModelPath string) error {
	onnxruntime.Init()
	v, err := onnxruntime.GetVersion()
	println("APIVersion", v)

	return err

}
