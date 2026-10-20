param(
    [Parameter(Mandatory = $true)][string]$Synthetic,
    [Parameter(Mandatory = $true)][string]$RealNoisy,
    [Parameter(Mandatory = $true)][string]$RealSpeech,
    [int]$EmbeddedFrames = 40
)

# One-shot M4 verification loop:
#   1. native scalar replay against the reference dumps,
#   2. native scalar oracle dumps (flat CFD1, outputs + 29 intermediates),
#   3. wasm build, instantiate in Node and compare every value bit for bit,
#      including a second pass through the embedded model.
#
# The reference paths are supplied by the caller; they are test data, not part
# of the engine or this script.

$ErrorActionPreference = "Stop"
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    Write-Host "== build =="
    cargo build --release -p cf-cli
    if ($LASTEXITCODE -ne 0) { throw "cf-cli build failed" }
    cargo build --release --target wasm32-unknown-unknown -p cf-wasm
    if ($LASTEXITCODE -ne 0) { throw "cf-wasm build failed" }

    $wasm = Join-Path (Get-Location) "target/wasm32-unknown-unknown/release/cf_wasm.wasm"
    $cli = Join-Path (Get-Location) "target/release/cf-cli.exe"
    $art = Join-Path (Get-Location) "artifacts/m4"
    New-Item -ItemType Directory -Force $art | Out-Null
    $wasmItem = Get-Item $wasm
    Write-Host ("wasm bytes: {0}" -f $wasmItem.Length)

    Write-Host "== native scalar replay vs reference =="
    foreach ($ref in @($Synthetic, $RealNoisy, $RealSpeech)) {
        & $cli check voice $ref --backend scalar
        if ($LASTEXITCODE -ne 0) { throw "native scalar check failed for $ref" }
    }

    Write-Host "== native scalar oracle dumps =="
    & $cli dump voice $Synthetic --out (Join-Path $art "synthetic.cfd") --layers
    if ($LASTEXITCODE -ne 0) { throw "synthetic oracle dump failed" }
    & $cli dump voice $RealNoisy --out (Join-Path $art "real-noisy.cfd")
    if ($LASTEXITCODE -ne 0) { throw "real-audio oracle dump failed" }
    & $cli dump voice $RealSpeech --out (Join-Path $art "real-speech.cfd")
    if ($LASTEXITCODE -ne 0) { throw "real-audio oracle dump failed" }

    Write-Host "== wasm vs native scalar oracle =="
    node crates/cf-wasm/harness/run.mjs `
        --wasm $wasm `
        --dump (Join-Path $art "synthetic.cfd") `
        --dump (Join-Path $art "real-noisy.cfd") `
        --dump (Join-Path $art "real-speech.cfd") `
        --embedded-check $EmbeddedFrames
    if ($LASTEXITCODE -ne 0) { throw "wasm harness failed" }
}
finally {
    Pop-Location
}
