package audio

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func generate(t *testing.T, args ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "source.bin")
	cmd := exec.Command("ffmpeg", append([]string{"-v", "error", "-y"}, append(args, p)...)...)
	if b, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("fixture generation: %v %s", e, b)
	}
	return p
}
func TestFormats(t *testing.T) {
	for _, format := range []string{"wav", "mp3", "flac", "ogg", "mp4", "webm", "adts"} {
		t.Run(format, func(t *testing.T) {
			args := []string{"-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-ar", "44100", "-ac", "2"}
			switch format {
			case "ogg":
				args = append(args, "-c:a", "libopus", "-ar", "48000")
			case "webm":
				args = append(args, "-c:a", "libopus", "-ar", "48000")
			case "mp4", "adts":
				args = append(args, "-c:a", "aac")
			}
			args = append(args, "-f", format)
			p := generate(t, args...)
			w, e := Prepare(context.Background(), filepath.Dir(p), 0)
			if e != nil {
				t.Fatal(e)
			}
			defer w.Close()
			if w.Samples < 1000 || w.Samples > 16000 {
				t.Fatal(w.Samples)
			}
			x, e := w.ReadSamples(0, 512)
			if e != nil || len(x) != 512 {
				t.Fatal(e, len(x))
			}
		})
	}
}
func TestNoAudio(t *testing.T) {
	p := generate(t, "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.1", "-an", "-c:v", "mpeg4", "-f", "mp4")
	if e := Probe(context.Background(), p); !errors.Is(e, ErrNoAudio) {
		t.Fatal(e)
	}

}
func TestPlaylistRejected(t *testing.T) {
	p := filepath.Join(t.TempDir(), "source.bin")
	_ = os.WriteFile(p, []byte("#EXTM3U\n#EXTINF:10,\nhttp://127.0.0.1/private.wav\n"), 0600)
	if e := Probe(context.Background(), p); !errors.Is(e, ErrInvalid) {
		t.Fatal(e)
	}
}
func TestCancelledConversion(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, e := command(ctx, "ffmpeg", "-version"); !errors.Is(e, context.Canceled) {
		t.Fatal(e)
	}
}
func TestMalformedWAV(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.wav")
	_ = os.WriteFile(p, []byte("RIFF\x00\x00\x00\x00WAVEdata\xff\xff\xff\xff"), 0600)
	if _, e := OpenWAV(p); e == nil {
		t.Fatal("invalid WAV accepted")
	}
}

// Sparse fixtures exercise large durations and RF64 without allocating huge buffers.
func sparseWAV(t *testing.T, seconds int64, rf64 bool) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "source.bin")
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	dataBytes := uint64(seconds * Rate * 2)
	header := make([]byte, 44)
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(dataBytes+36))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], Rate)
	binary.LittleEndian.PutUint32(header[28:], Rate*2)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(dataBytes))
	if rf64 {
		copy(header, "RF64")
		binary.LittleEndian.PutUint32(header[4:], 0xffffffff)
		binary.LittleEndian.PutUint32(header[40:], 0xffffffff)
		ds := make([]byte, 36)
		copy(ds, "ds64")
		binary.LittleEndian.PutUint32(ds[4:], 28)
		binary.LittleEndian.PutUint64(ds[8:], dataBytes+72)
		binary.LittleEndian.PutUint64(ds[16:], dataBytes)
		binary.LittleEndian.PutUint64(ds[24:], dataBytes/2)
		header = append(append(append([]byte{}, header[:12]...), ds...), header[12:]...)
	}
	if _, err = f.Write(header); err != nil {
		t.Fatal(err)
	}
	if err = f.Truncate(int64(len(header)) + int64(dataBytes)); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRecordingBeyondFourHours(t *testing.T) {
	const seconds = 4*60*60 + 1
	p := sparseWAV(t, seconds, false)
	w, err := Prepare(context.Background(), filepath.Dir(p), 0)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if w.Samples != seconds*Rate {
		t.Fatalf("recording truncated: %d samples", w.Samples)
	}
}

func TestRF64BeyondFourGiB(t *testing.T) {
	const seconds = 40 * 60 * 60
	p := sparseWAV(t, seconds, true)
	w, err := OpenWAV(p)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if w.Samples != seconds*Rate {
		t.Fatal(w.Samples)
	}
	x, err := w.ReadSamples(w.Samples-512, 512)
	if err != nil || len(x) != 512 {
		t.Fatal(err, len(x))
	}
	if err := os.Truncate(p, 80); err != nil {
		t.Fatal(err)
	}
	if bad, err := OpenWAV(p); err == nil {
		bad.Close()
		t.Fatal("truncated RF64 accepted")
	}
}

func TestConversionDiskLimitRejectsPartialAudio(t *testing.T) {
	p := generate(t, "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-f", "wav")
	err := convert(context.Background(), p, filepath.Join(filepath.Dir(p), "limited.wav"), 8192)
	if !errors.Is(err, ErrStorage) {
		t.Fatalf("partial audio must not be transcribed: %v", err)
	}
}
