# Build the native CleanField library and verify both link modes.
#
#   pwsh -File scripts/verify_m6_link.ps1
#
# The example is a plain Rust consumer that declares the C ABI by hand (it
# stands in for a C translation unit).  It is compiled once against the import
# library (dynamic) and once against the static archive, then run.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    cargo build --release -p cf-ffi
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

    $out = Join-Path $root "artifacts\m6"
    New-Item -ItemType Directory -Force $out | Out-Null

    rustc -O --edition 2021 examples\native_embed.rs `
        -L target\release -l dylib=cleanfield -o "$out\embed-dyn.exe"
    if ($LASTEXITCODE -ne 0) { throw "dynamic link failed" }

    Copy-Item target\release\cleanfield.dll "$out\cleanfield.dll" -Force
    & "$out\embed-dyn.exe"
    if ($LASTEXITCODE -ne 0) { throw "dynamic example failed" }

    rustc -O --edition 2021 examples\native_embed.rs `
        -L target\release -l static=cleanfield -o "$out\embed-static.exe"
    if ($LASTEXITCODE -ne 0) { throw "static link failed" }

    & "$out\embed-static.exe"
    if ($LASTEXITCODE -ne 0) { throw "static example failed" }

    Write-Host "OK: dynamic and static ABI link examples passed"
} finally {
    Pop-Location
}
