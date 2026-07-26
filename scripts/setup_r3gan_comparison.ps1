[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $ProjectRoot.StartsWith("D:\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "This comparison setup is restricted to D:. Current project: $ProjectRoot"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found at $Python. Create .venv and run scripts\\setup_local_gpu.ps1 first."
}

$TmpRoot = Join-Path $ProjectRoot "tmp"
$SourceDir = Join-Path $ProjectRoot "third_party\R3GAN"
$ModelDir = Join-Path $ProjectRoot "models\r3gan"
$env:PIP_CACHE_DIR = Join-Path $TmpRoot "pip-cache"
$env:TEMP = $TmpRoot
$env:TMP = $TmpRoot
$env:HF_HOME = Join-Path $TmpRoot "huggingface"
$env:TORCH_EXTENSIONS_DIR = Join-Path $TmpRoot "r3gan-torch-extensions"
$env:DNNLIB_CACHE_DIR = Join-Path $TmpRoot "r3gan-dnnlib-cache"

@($TmpRoot, $env:PIP_CACHE_DIR, $env:HF_HOME, $env:TORCH_EXTENSIONS_DIR, $env:DNNLIB_CACHE_DIR, $ModelDir, (Split-Path -Parent $SourceDir)) |
    ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }

if (-not (Test-Path -LiteralPath (Join-Path $SourceDir "legacy.py"))) {
    if (Test-Path -LiteralPath $SourceDir) {
        throw "Refusing to overwrite the existing directory $SourceDir. Move it aside after reviewing it, then rerun this script."
    }
    Write-Host "Cloning official R3GAN source into D:..."
    git clone --depth 1 https://github.com/brownvc/R3GAN.git $SourceDir
} else {
    Write-Host "Using existing R3GAN source at $SourceDir"
}

Write-Host "Installing the small Python dependencies with caches on D:..."
# R3GAN's upstream StyleGAN-derived operators still import pkg_resources.
# setuptools removed it in v82, while the project's CUDA PyTorch wheel requires
# setuptools >=77, so keep the compatible interval explicit.
& $Python -m pip install --cache-dir $env:PIP_CACHE_DIR "setuptools>=77,<82" "huggingface_hub>=0.24,<1" "click>=8.1,<9" "requests>=2.31,<3" "scipy>=1.11,<2" "ninja>=1.11,<2"
if ($LASTEXITCODE -ne 0) { throw "Python dependency installation failed." }

$Checkpoint = Join-Path $ModelDir "network-snapshot-final.pkl"
if (-not (Test-Path -LiteralPath $Checkpoint)) {
    Write-Host "Downloading the official 645 MB R3GAN FFHQ-256 checkpoint into D:..."
    $env:FACEFORGE_R3GAN_DOWNLOAD_DIR = $ModelDir
    $DownloadCode = @'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

target = Path(os.environ["FACEFORGE_R3GAN_DOWNLOAD_DIR"])
target.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="brownvc/R3GAN-FFHQ-256x256",
    filename="network-snapshot-final.pkl",
    local_dir=target,
)
print(path)
'@
    # PowerShell's native-command argument parsing can remove quotes from a
    # multiline value passed directly to ``python -c``.  Keep the source in an
    # environment variable and pass only a quote-safe launcher instead.
    $env:FACEFORGE_R3GAN_DOWNLOAD_CODE = $DownloadCode
    $PythonLauncher = "import os; exec(compile(os.environ['FACEFORGE_R3GAN_DOWNLOAD_CODE'], '<r3gan-download>', 'exec'))"
    & $Python -c $PythonLauncher
    if ($LASTEXITCODE -ne 0) { throw "Official R3GAN checkpoint download failed." }
} else {
    Write-Host "Using existing official checkpoint at $Checkpoint"
}

Write-Host ""
Write-Host "R3GAN comparison assets are installed on D:."
Write-Host "The app uses the upstream PyTorch reference operators by default, so a separate CUDA Toolkit is not required for inference."
Write-Host "Set FACEFORGE_R3GAN_CUSTOM_OPS=1 only if you intentionally install Visual Studio C++ Build Tools and the CUDA Toolkit to compile optional faster extensions."
Write-Host "Only load the official checkpoint downloaded by this script: R3GAN uses a Python pickle, and untrusted pickles are unsafe."
