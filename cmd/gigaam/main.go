package main

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/api"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/inference"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
)

func main() {
	if e := run(); e != nil {
		logging.Log.Error().Str("reason", e.Error()).Msg("service stopped")
		os.Exit(1)
	}
}
func run() error {
	c, e := config.Load()
	if e != nil {
		return e
	}
	service := api.New(c)
	listener, e := net.Listen("tcp", ":"+c.Port)
	if e != nil {
		return fmt.Errorf("cannot listen on configured port")
	}
	server := &http.Server{Handler: service.Handler(), ReadHeaderTimeout: 15 * time.Second, IdleTimeout: time.Minute, MaxHeaderBytes: 16 << 10}
	served := make(chan error, 1)
	go func() { served <- server.Serve(listener) }()
	logging.Log.Info().Str("port", c.Port).Str("model", c.Model).Msg("HTTP listening")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go service.RetryCleanup(ctx)
	var engine *inference.Engine
	var init sync.WaitGroup
	init.Add(1)
	go func() {
		defer init.Done()
		engine = initialize(ctx, c, service)
	}()
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGTERM, syscall.SIGINT)
	defer signal.Stop(sig)
	var serveErr error
	select {
	case <-sig:
		logging.Log.Info().Str("stage", "draining").Msg("shutdown")
	case err := <-served:
		if err != nil && err != http.ErrServerClosed {
			serveErr = fmt.Errorf("HTTP server failed")
		}
	}
	grace, stopGrace := context.WithTimeout(context.Background(), c.GracePeriod)
	service.Drain(grace)
	stopGrace()
	cancel()
	init.Wait()
	shutdown, stop := context.WithTimeout(context.Background(), c.ResponseTimeout)
	defer stop()
	if err := server.Shutdown(shutdown); err != nil {
		_ = server.Close()
	}
	if engine != nil {
		engine.Close()
	}
	return serveErr
}
