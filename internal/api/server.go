package api

import (
	"context"
	"os"
	"sync"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/audio"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
)

type Engine interface {
	Transcribe(context.Context, *audio.WAV) (string, error)
}

type Server struct {
	c         config.Config
	mu        sync.Mutex
	state     string
	stopping  bool
	active    int
	drained   chan struct{}
	engine    Engine
	slots     chan struct{}
	pending   map[string]bool
	removeAll func(string) error
	freeBytes func(string) (uint64, error)
	root      context.Context
	cancel    context.CancelFunc
}

func New(c config.Config) *Server {
	ctx, cancel := context.WithCancel(context.Background())
	return &Server{
		c:         c,
		state:     "service_initializing",
		slots:     make(chan struct{}, c.Concurrency),
		pending:   make(map[string]bool),
		removeAll: os.RemoveAll,
		freeBytes: audio.FreeBytes,
		root:      ctx,
		cancel:    cancel,
		drained:   make(chan struct{}),
	}
}
