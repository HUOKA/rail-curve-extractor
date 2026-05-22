param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $RepoRoot ".open3d-venv"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -m venv $VenvDir
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install -U pip
& $VenvPython -m pip install "numpy>=2.1" "laspy[lazrs]>=2.5" "open3d>=0.19"

Write-Host "Open3D viewer environment ready: $VenvPython"

