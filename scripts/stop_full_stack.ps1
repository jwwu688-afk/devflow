# Stop backend / frontend processes started by start_full_stack.ps1.
# Docker containers are kept by default; pass -StopDocker to stop them too.

param(
    [switch]$StopDocker
)

$ErrorActionPreference = "Continue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $projectRoot ".data"

function Stop-FromPidFile($pidFile, $label) {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        Write-Host "[devflow] $label pid file not found, skipped." -ForegroundColor Yellow
        return
    }
    $processId = (Get-Content -LiteralPath $pidFile | Select-Object -First 1).Trim()
    if (-not $processId) { return }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) {
        # next dev spawns child node processes; kill the whole tree
        & taskkill /PID $processId /T /F 2>&1 | Out-Null
        Write-Host "[devflow] stopped $label (PID $processId)" -ForegroundColor Cyan
    } else {
        Write-Host "[devflow] $label (PID $processId) already gone." -ForegroundColor Yellow
    }
    Remove-Item -LiteralPath $pidFile -Force
}

Stop-FromPidFile (Join-Path $dataDir "devflow-frontend.pid") "frontend"
Stop-FromPidFile (Join-Path $dataDir "devflow-backend.pid") "backend"

if ($StopDocker) {
    Push-Location $projectRoot
    try {
        docker compose stop postgres etcd minio milvus | Out-Host
    } finally {
        Pop-Location
    }
}

Write-Host "[devflow] done." -ForegroundColor Cyan
