Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\\Scripts\\python.exe"

if (-not (Test-Path $Python)) {
    throw "python not found: $Python"
}

Push-Location $Root
try {
    & $Python ".\\scripts\\train_kobart_black_probe_repair.py"
}
finally {
    Pop-Location
}
