$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ((Split-Path -Qualifier $ProjectRoot) -ne "D:") {
    throw "This setup must run from partition D:. Current project root: $ProjectRoot"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found at $Python. Create it on D: first."
}

$Tmp = Join-Path $ProjectRoot "tmp"
$PipCache = Join-Path $Tmp "pip-cache"
New-Item -ItemType Directory -Force -Path $Tmp, $PipCache | Out-Null
$env:TEMP = $Tmp
$env:TMP = $Tmp
$env:PIP_CACHE_DIR = $PipCache
$env:TORCH_HOME = (Join-Path $Tmp "torch")

# CUDA 12.6 wheels are compatible with the installed NVIDIA driver and avoid the CPU-only build.
# Do not reinstall dependencies here: Windows can fail while rolling back an unrelated
# package even after the 2.6 GB CUDA wheel has downloaded successfully.
& $Python -m pip install --upgrade --force-reinstall --no-deps `
    --index-url https://download.pytorch.org/whl/cu126 torch torchvision
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -c "import torch; assert torch.cuda.is_available(), 'CUDA was not detected'; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
