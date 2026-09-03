$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot '.venv'
$VenvPython = Join-Path $VenvPath 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $VenvPython)) {
    # Reuse the already validated CUDA-enabled torch installation read-only.
    # Packages installed from this environment are still written under .venv.
    python -m venv --system-site-packages $VenvPath
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install pytest==8.4.2
& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA-enabled torch is required'; print(torch.__version__, torch.cuda.get_device_name(0))"
& $VenvPython (Join-Path $ProjectRoot 'audit.py') --stage all
