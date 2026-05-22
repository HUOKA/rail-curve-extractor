param(
    [string]$PythonExe = ".\.venv\Scripts\python.exe",
    [string]$Version = "dev",
    [string]$Platform = "win64"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

& $PythonExe -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) {
    throw "Unit tests failed."
}

& $PythonExe -m PyInstaller --noconfirm --clean packaging\rail_curve_extractor.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$distRoot = Join-Path $repoRoot "dist"
$baseDir = Join-Path $distRoot "RailCurveExtractor"
if (-not (Test-Path $baseDir)) {
    throw "Expected portable directory not found: $baseDir"
}

$versionedDirName = "RailCurveExtractor-$Platform-$Version"
$portableDir = Join-Path $distRoot $versionedDirName
if (Test-Path $portableDir) {
    Remove-Item $portableDir -Recurse -Force
}
Move-Item $baseDir $portableDir

$baseExe = Join-Path $portableDir "RailCurveExtractor.exe"
$versionedExeName = "RailCurveExtractor-$Version.exe"
$versionedExe = Join-Path $portableDir $versionedExeName
if (Test-Path $baseExe) {
    Rename-Item $baseExe -NewName $versionedExeName
}

switch ($Platform) {
    "win64" {
        $archiveName = "RailCurveExtractor-$Platform-portable-$Version.zip"
        $archivePath = Join-Path $distRoot $archiveName
        if (Test-Path $archivePath) {
            Remove-Item $archivePath -Force
        }
        Compress-Archive -Path (Join-Path $portableDir "*") -DestinationPath $archivePath -Force
    }
    default {
        throw "Unsupported platform in build_portable.ps1: $Platform"
    }
}

Write-Host "Portable directory:  $portableDir"
Write-Host "Portable executable: $versionedExe"
Write-Host "Portable archive:    $archivePath"
