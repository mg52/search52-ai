# syntax=docker/dockerfile:1

# Pin to the toolchain declared in go.mod; override at build time if you bump it:
#   docker build --build-arg GO_VERSION=1.26.4 -t search52-ai .
ARG GO_VERSION=1.26.4

# ---------- build stage ----------
FROM golang:${GO_VERSION}-alpine AS build
WORKDIR /src

# Resolve modules first so this layer is cached unless go.mod changes.
COPY go.mod ./
RUN go mod download

COPY . .

# Fully static, stripped binary so it runs on a scratch/distroless image.
# BuildKit cache mounts keep CI builds fast.
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg/mod \
    CGO_ENABLED=0 GOOS=linux GOFLAGS=-trimpath \
    go build -ldflags "-s -w" -o /out/server ./cmd/server

# ---------- runtime stage ----------
# distroless/static:nonroot ships CA certificates (needed for outbound HTTPS to
# the LLM/embedding APIs), runs as an unprivileged user, and has no shell or
# package manager to attack.
FROM gcr.io/distroless/static:nonroot
WORKDIR /app

COPY --from=build /out/server /app/server
COPY --from=build /src/prompts /app/prompts

ENV PORT=8080 \
    PROMPTS_DIR=/app/prompts

EXPOSE 8080
USER nonroot:nonroot

# Orchestrators should probe GET /health for liveness/readiness (the distroless
# image has no shell/curl for a Docker HEALTHCHECK).
ENTRYPOINT ["/app/server"]
