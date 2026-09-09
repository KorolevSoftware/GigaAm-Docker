.PHONY: build test vet fmt integration

build:
	go build -trimpath -o bin/gigaam ./cmd/gigaam

test:
	go test -race ./...

vet:
	go vet ./...

fmt:
	gofmt -w cmd internal

integration:
	@test -n "$(GIGAAM_TEST_MODELS)" -a -n "$(GIGAAM_TEST_ORT)" -a -n "$(GIGAAM_TEST_GOLDEN)" || (echo 'Set GIGAAM_TEST_MODELS, GIGAAM_TEST_ORT, GIGAAM_TEST_GOLDEN'; exit 1)
	go test -v ./internal/inference -run TestGolden -count=1 -timeout=30m
