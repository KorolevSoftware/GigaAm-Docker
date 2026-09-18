package audio

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
)

var ErrInvalid = errors.New("invalid audio")
var ErrNoAudio = errors.New("no audio stream")
var ErrUnavailable = errors.New("media tool unavailable")
var ErrStorage = errors.New("media storage unavailable")

const Rate = 16000
const formats = "wav,mp3,flac,mov,matroska,webm,ogg,aac"

type boundedBuffer struct {
	data  []byte
	limit int
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	n := len(p)
	if left := b.limit - len(b.data); left > 0 {
		if len(p) > left {
			p = p[:left]
		}
		b.data = append(b.data, p...)
	}
	return n, nil
}
func command(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var out = boundedBuffer{limit: 64 << 10}
	var stderr = boundedBuffer{limit: 4096}
	cmd.Stdout = &out
	cmd.Stderr = &stderr
	e := cmd.Run()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if e != nil {
		if errors.Is(e, exec.ErrNotFound) {
			return nil, ErrUnavailable
		}
		if strings.Contains(string(stderr.data), "No space left on device") {
			return nil, ErrStorage
		}
		return nil, ErrInvalid
	}
	return out.data, nil
}
func Probe(ctx context.Context, path string) error {
	data, e := command(ctx, "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", formats, "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "json", path)
	if e != nil {
		return e
	}
	var p struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
		} `json:"streams"`
	}
	if json.Unmarshal(data, &p) != nil {
		return ErrInvalid
	}
	if len(p.Streams) == 0 {
		return ErrNoAudio
	}
	return nil
}
func Prepare(ctx context.Context, dir string, reserveBytes int64) (*WAV, error) {
	src := filepath.Join(dir, "source.bin")
	finishProbe := metrics.Start(ctx, metrics.Probe)
	probeErr := Probe(ctx, src)
	finishProbe(metrics.Code(probeErr))
	if e := probeErr; e != nil {
		return nil, e
	}
	dst := filepath.Join(dir, "prepared.wav")
	// Bound temporary output by available disk, not recording duration. FFmpeg
	// can finish successfully at -fs, so reaching the bound must be an error.
	free, e := FreeBytes(dir)
	if e != nil || reserveBytes < 0 || free <= uint64(reserveBytes)+4096 {
		return nil, ErrStorage
	}
	outputLimit := min(free-uint64(reserveBytes), uint64(math.MaxInt64))
	finishConvert := metrics.Start(ctx, metrics.Convert)
	e = convert(ctx, src, dst, outputLimit)
	finishConvert(metrics.Code(e))
	if e != nil {
		return nil, e
	}
	return OpenWAV(dst)
}

func convert(ctx context.Context, src, dst string, outputLimit uint64) error {
	_, e := command(ctx, "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-protocol_whitelist", "file,pipe", "-format_whitelist", formats, "-i", src, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-rf64", "auto", "-fs", strconv.FormatUint(outputLimit, 10), dst)
	if e == nil {
		var st os.FileInfo
		st, e = os.Stat(dst)
		if e == nil && uint64(st.Size()) >= outputLimit {
			e = ErrStorage
		}
	}
	return e
}

type WAV struct {
	file    *os.File
	offset  int64
	Samples int64
}

func OpenWAV(path string) (*WAV, error) {
	f, e := os.Open(path)
	if e != nil {
		return nil, e
	}
	ok := false
	defer func() {
		if !ok {
			f.Close()
		}
	}()
	var h [12]byte
	if _, e = io.ReadFull(f, h[:]); e != nil || (string(h[:4]) != "RIFF" && string(h[:4]) != "RF64") || string(h[8:]) != "WAVE" {
		return nil, ErrInvalid
	}
	st, e := f.Stat()
	if e != nil {
		return nil, e
	}
	valid := false
	rf64 := string(h[:4]) == "RF64"
	var dataSize uint64
	hasDataSize := false
	for {
		var ch [8]byte
		if _, e = io.ReadFull(f, ch[:]); e != nil {
			return nil, ErrInvalid
		}
		n := int64(binary.LittleEndian.Uint32(ch[4:]))
		off, _ := f.Seek(0, io.SeekCurrent)
		if rf64 && string(ch[:4]) == "data" && n == 0xffffffff {
			if !hasDataSize || dataSize > uint64(st.Size()-off) {
				return nil, ErrInvalid
			}
			n = int64(dataSize)
		}
		if n > st.Size()-off {
			return nil, ErrInvalid
		}
		switch string(ch[:4]) {
		case "ds64":
			if !rf64 || hasDataSize || n < 28 {
				return nil, ErrInvalid
			}
			var b [28]byte
			if _, e = io.ReadFull(f, b[:]); e != nil {
				return nil, ErrInvalid
			}
			dataSize = binary.LittleEndian.Uint64(b[8:16])
			hasDataSize = true
		case "fmt ":
			if n < 16 {
				return nil, ErrInvalid
			}
			var b [16]byte
			if _, e = io.ReadFull(f, b[:]); e != nil {
				return nil, ErrInvalid
			}
			valid = binary.LittleEndian.Uint16(b[0:]) == 1 && binary.LittleEndian.Uint16(b[2:]) == 1 && binary.LittleEndian.Uint32(b[4:]) == Rate && binary.LittleEndian.Uint16(b[12:]) == 2 && binary.LittleEndian.Uint16(b[14:]) == 16
		case "data":
			if !valid || n%2 != 0 {
				return nil, ErrInvalid
			}
			ok = true
			return &WAV{f, off, n / 2}, nil
		}
		if _, e = f.Seek(off+n+n%2, io.SeekStart); e != nil {
			return nil, e
		}
	}
}
func (w *WAV) Close() error { return w.file.Close() }
func (w *WAV) ReadSamples(start int64, count int) ([]float32, error) {
	if start < 0 || start > w.Samples || count < 0 {
		return nil, ErrInvalid
	}
	if int64(count) > w.Samples-start {
		count = int(w.Samples - start)
	}
	b := make([]byte, count*2)
	if _, e := w.file.ReadAt(b, w.offset+start*2); e != nil {
		return nil, e
	}
	x := make([]float32, count)
	for i := range x {
		x[i] = float32(int16(binary.LittleEndian.Uint16(b[i*2:]))) / 32768
	}
	return x, nil
}
