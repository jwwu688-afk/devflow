# DevFlow AI - full stack launcher (PostgreSQL + Milvus + real embedding)
#
# Unlike start_rag.ps1 (SQLite + deterministic embedding, deliberately degraded),
# this script starts the complete stack:
#   docker: postgres(pgvector) + etcd + minio + milvus
#   backend: uvicorn with real Postgres/Milvus and the configured embedding provider
#   frontend: next dev
#
# Usage:
#   .\scripts\start_full_stack.ps1
#   .\scripts\start_full_stack.ps1 -SkipDocker        # infra already running
#   .\scripts\start_full_stack.ps1 -BackendPort 8001 -FrontendPort 3001
#
# Stop with: .\scripts\stop_full_stack.ps1

param(
    [string]$PythonPath = "D:\Anaconda\envs\devflow\python.exe",
    [ValidateRange(1, 65535)][int]$BackendPort = 8000,
    [ValidateRange(1, 65535)][int]$FrontendPort = 3000,
    [string]$HfEndpoint = "https://hf-mirror.com",
    [switch]$SkipDocker,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $projectRoot "backend"
$frontendDir = Join-Path $projectRoot "frontend"
$dataDir = Join-Path $projectRoot ".data"

if (-not (Test-Path -LiteralPath $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force | Out-Null
}

function Write-Step($message) { Write-Host "[devflow] $message" -ForegroundColor Cyan }
function Write-Warn($message) { Write-Host "[devflow] $message" -ForegroundColor Yellow }
function Write-Fail($message) { Write-Host "[devflow] $message" -ForegroundColor Red }

function Wait-Http($url, $timeoutSeconds, $label) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                Write-Step "$label is up ($url)"
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 800
        }
    }
    Write-Fail "$label did not become ready within $timeoutSeconds s ($url)"
    return $false
}

# --- preflight ---------------------------------------------------------------
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    Write-Fail "Python interpreter not found: $PythonPath"
    Write-Fail "Pass -PythonPath <path to your python.exe> (conda env or venv)."
    exit 1
}

$envFile = Join-Path $projectRoot ".env"
if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    Copy-Item (Join-Path $projectRoot ".env.example") $envFile
    Write-Warn ".env was missing, copied from .env.example. Review it before real use."
}

# --- infrastructure ----------------------------------------------------------
if (-not $SkipDocker) {
    Write-Step "Checking Docker daemon..."
    docker info --format "{{.ServerVersion}}" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Docker daemon is not reachable. Start Docker Desktop first, then rerun."
        exit 1
    }

    Write-Step "Starting postgres / etcd / minio / milvus..."
    Push-Location $projectRoot
    try {
        docker compose up -d postgres etcd minio milvus | Out-Host
        if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up failed."; exit 1 }
    } finally {
        Pop-Location
    }

    Write-Step "Waiting for PostgreSQL..."
    $pgDeadline = (Get-Date).AddSeconds(90)
    $pgReady = $false
    while ((Get-Date) -lt $pgDeadline) {
        docker exec devflow-ai-postgres pg_isready -U devflow -d devflow 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $pgReady = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $pgReady) { Write-Fail "PostgreSQL did not become ready."; exit 1 }
    Write-Step "PostgreSQL is ready."

    Write-Step "Waiting for Milvus..."
    if (-not (Wait-Http "http://127.0.0.1:9091/healthz" 180 "Milvus")) { exit 1 }
}
else {
    Write-Warn "Skipping docker startup (-SkipDocker)."
}

# --- backend -----------------------------------------------------------------
# The local Qwen3 embedding model is pulled from the Hugging Face hub on first
# use. huggingface.co is usually unreachable from mainland China, so point the
# client at a mirror unless the caller already set HF_ENDPOINT.
if (-not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = $HfEndpoint
    Write-Step "HF_ENDPOINT=$($env:HF_ENDPOINT) (override with -HfEndpoint or set it yourself)"
}
# The mirror does not proxy Xet storage; without this the download dies with a
# 401 from cas-server.xethub.hf.co.
if (-not $env:HF_HUB_DISABLE_XET) {
    $env:HF_HUB_DISABLE_XET = "1"
}

Write-Step "Starting backend on port $BackendPort..."
$backendOut = Join-Path $dataDir "devflow-backend.out.log"
$backendErr = Join-Path $dataDir "devflow-backend.err.log"

$backendProcess = Start-Process -FilePath $PythonPath `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
    -WorkingDirectory $backendDir `
    -RedirectStandardOutput $backendOut `
    -RedirectStandardError $backendErr `
    -PassThru -WindowStyle Hidden

$pidFile = Join-Path $dataDir "devflow-backend.pid"
Set-Content -LiteralPath $pidFile -Value $backendProcess.Id -Encoding ascii
Write-Step "Backend PID $($backendProcess.Id) (log: $backendErr)"

if (-not (Wait-Http "http://127.0.0.1:$BackendPort/health" 120 "Backend")) {
    Write-Fail "Backend failed to start. Last 40 lines of $backendErr :"
    if (Test-Path -LiteralPath $backendErr) { Get-Content -LiteralPath $backendErr -Tail 40 | Out-Host }
    exit 1
}

# --- frontend ----------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Step "Starting frontend on port $FrontendPort..."
    $frontendOut = Join-Path $dataDir "devflow-frontend.out.log"
    $frontendErr = Join-Path $dataDir "devflow-frontend.err.log"

    $frontendProcess = Start-Process -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "-p", "$FrontendPort") `
        -WorkingDirectory $frontendDir `
        -RedirectStandardOutput $frontendOut `
        -RedirectStandardError $frontendErr `
        -PassThru -WindowStyle Hidden

    $frontendPidFile = Join-Path $dataDir "devflow-frontend.pid"
    Set-Content -LiteralPath $frontendPidFile -Value $frontendProcess.Id -Encoding ascii
    Write-Step "Frontend PID $($frontendProcess.Id) (log: $frontendErr)"

    if (-not (Wait-Http "http://127.0.0.1:$FrontendPort" 180 "Frontend")) {
        Write-Fail "Frontend did not become ready. Check $frontendErr"
    }
}

Write-Host ""
Write-Step "DevFlow AI is up."
Write-Host "  Backend  : http://127.0.0.1:$BackendPort"
Write-Host "  API docs : http://127.0.0.1:$BackendPort/docs"
if (-not $SkipFrontend) {
    Write-Host "  Frontend : http://127.0.0.1:$FrontendPort"
}
Write-Host "  Logs     : $dataDir"
