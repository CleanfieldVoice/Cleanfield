param(
    [Parameter(Mandatory = $true)][string]$Reference,
    [string[]]$Synthetic = @(),
    [string[]]$RealAudio = @(),
    [int]$Cases = 10000000,
    [int]$ScalarFrames = 3000,
    [int]$AvxFrames = 10000
)

# One-shot M3 verification loop: both backends against the reference dumps,
# every intermediate of both backends against each other, then the benchmark.
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

    foreach ($backend in @("scalar", "avx2")) {
        Write-Host "== check voice ($backend) $Reference =="
        cargo run --release -q -p cf-cli -- check voice $Reference --backend $backend
        if ($LASTEXITCODE -ne 0) { throw "check failed for $Reference ($backend)" }

        foreach ($dump in $Synthetic) {
            Write-Host "== check voice ($backend) $dump =="
            cargo run --release -q -p cf-cli -- check voice $dump --backend $backend
            if ($LASTEXITCODE -ne 0) { throw "check failed for $dump ($backend)" }
        }
        foreach ($dump in $RealAudio) {
            Write-Host "== check voice ($backend, real audio) $dump =="
            cargo run --release -q -p cf-cli -- check voice $dump --backend $backend
            if ($LASTEXITCODE -ne 0) { throw "check failed for $dump ($backend)" }
        }
    }

    Write-Host "== simd-check voice $Reference =="
    cargo run --release -q -p cf-cli -- simd-check voice $Reference
    if ($LASTEXITCODE -ne 0) { throw "simd-check failed for $Reference" }
    foreach ($dump in $Synthetic) {
        Write-Host "== simd-check voice $dump =="
        cargo run --release -q -p cf-cli -- simd-check voice $dump
        if ($LASTEXITCODE -ne 0) { throw "simd-check failed for $dump" }
    }
    foreach ($dump in $RealAudio) {
        Write-Host "== simd-check voice (real audio) $dump =="
        cargo run --release -q -p cf-cli -- simd-check voice $dump
        if ($LASTEXITCODE -ne 0) { throw "simd-check failed for $dump" }
    }

    Write-Host "== bench voice (scalar) =="
    cargo run --release -q -p cf-cli -- bench voice $ScalarFrames --backend scalar
    if ($LASTEXITCODE -ne 0) { throw "scalar bench failed" }

    Write-Host "== bench voice (avx2) =="
    cargo run --release -q -p cf-cli -- bench voice $AvxFrames --backend avx2 --per-layer
    if ($LASTEXITCODE -ne 0) { throw "avx2 bench failed" }
}
finally {
    Pop-Location
}
