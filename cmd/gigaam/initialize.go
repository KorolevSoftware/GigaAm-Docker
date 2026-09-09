package main

import (
	"context"
	"time"

	"github.com/KorolevSoftware/GigaAm-Docker/internal/api"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/config"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/diagnostics"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/inference"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/logging"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/metrics"
	"github.com/KorolevSoftware/GigaAm-Docker/internal/models"
)

func initialize(ctx context.Context, c config.Config, service *api.Server) *inference.Engine {
	initStart := time.Now()
	if err := service.CleanStartup(); err != nil {
		service.Unavailable()
		logging.Log.Error().Str("stage", "startup_cleanup").Str("code", "storage_unavailable").Str("reason_code", diagnostics.Code(err)).Msg("initialization failed")
		return nil
	}
	setupStart := time.Now()
	logging.Log.Info().Str("stage", "model_setup").Str("event", "start").Msg("initialization")
	asr, vad, err := models.Locate(c.ModelDir, c.Model)
	logging.Log.Info().Str("stage", "model_setup").Str("event", "complete").Float64("duration_ms", float64(time.Since(setupStart))/float64(time.Millisecond)).Str("code", metrics.Code(err)).Msg("initialization")
	if err != nil {
		service.Unavailable()
		logging.Log.Error().Str("stage", "model_setup").Str("code", "model_unavailable").Str("reason_code", diagnostics.Code(err)).Msg("initialization failed")
		return nil
	}
	loadStart := time.Now()
	logging.Log.Info().Str("stage", "model_load").Str("event", "start").Msg("initialization")
	engine, err := inference.New(ctx, c, asr, vad)
	logging.Log.Info().Str("stage", "model_load").Str("event", "complete").Float64("duration_ms", float64(time.Since(loadStart))/float64(time.Millisecond)).Str("code", metrics.Code(err)).Msg("initialization")
	if err != nil {
		service.Unavailable()
		logging.Log.Error().Str("stage", "inference").Str("code", "model_unavailable").Str("reason_code", diagnostics.Code(err)).Msg("initialization failed")
		return nil
	}
	service.Ready(engine)
	logging.Log.Info().Str("model", c.Model).Float64("duration_ms", float64(time.Since(initStart))/float64(time.Millisecond)).Msg("model ready")
	return engine
}
