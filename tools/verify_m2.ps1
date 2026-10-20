param(
    [Parameter(Mandatory = $true)][string]$Reference,
    [string[]]$RealAudio = @(),
    [int]$Cases = 10000000,
    [int]$Frames = 2000
)

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Write-Host "== cargo test --workspace =="
    cargo test --workspace
    if ($LASTEXITCODE -ne 0) { throw "cargo test failed" }

    Write-Host "== fma-check =="
    cargo run --release -q -p cf-cli -- fma-check --cases $Cases
    if ($LASTEXITCODE -ne 0) { throw "fma-check failed" }

    Write-Host "== rcp-check =="
    cargo run --release -q -p cf-cli -- rcp-check
    if ($LASTEXITCODE -ne 0) { throw "rcp-check failed" }

    Write-Host "== check voice (synthetic) =="
    cargo run --release -q -p cf-cli -- check voice $Reference
    if ($LASTEXITCODE -ne 0) { throw "check failed for $Reference" }

    foreach ($dump in $RealAudio) {
        Write-Host "== check voice (real audio) $dump =="
        cargo run --release -q -p cf-cli -- check voice $dump
        if ($LASTEXITCODE -ne 0) { throw "check failed for $dump" }
    }

    Write-Host "== bench voice =="
    cargo run --release -q -p cf-cli -- bench voice $Frames
    if ($LASTEXITCODE -ne 0) { throw "bench failed" }
}
finally {
    Pop-Location
}
