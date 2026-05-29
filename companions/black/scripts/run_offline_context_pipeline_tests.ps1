$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$projectRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent (Split-Path -Parent $projectRoot)
Set-Location $workspaceRoot

python -m pytest companions\black\tests\test_black_offline_context_pipeline.py -q --tb=short -n auto
exit $LASTEXITCODE
