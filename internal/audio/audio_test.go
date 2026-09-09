package audio

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
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
			w, e := Prepare(context.Background(), filepath.Dir(p), time.Second)
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
func TestNoAudioAndDuration(t *testing.T) {
	p := generate(t, "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.1", "-an", "-c:v", "mpeg4", "-f", "mp4")
	if e := Probe(context.Background(), p, time.Second); !errors.Is(e, ErrNoAudio) {
		t.Fatal(e)
	}
	p = generate(t, "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "-f", "wav")
	if _, e := Prepare(context.Background(), filepath.Dir(p), time.Second); !errors.Is(e, ErrDuration) {
		t.Fatal(e)
	}
}
func TestPlaylistRejected(t *testing.T) {
	p := filepath.Join(t.TempDir(), "source.bin")
	_ = os.WriteFile(p, []byte("#EXTM3U\n#EXTINF:10,\nhttp://127.0.0.1/private.wav\n"), 0600)
	if e := Probe(context.Background(), p, time.Second); !errors.Is(e, ErrInvalid) {
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
