# One-shot setup for a Windows box, PowerShell version.
#   Right-click > Run with PowerShell
#   or:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
# If PowerShell refuses to run it, either use the .bat version or run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Fail($msg) {
    Write-Host "[x] $msg" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    Fail "The Python launcher 'py' was not found. Install 64-bit Python 3.11 from python.org and tick 'Add python.exe to PATH'."
}

Write-Host "[1/4] creating .venv"
try { py -3.11 -m venv .venv } catch { }
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3 -m venv .venv }
if (-not (Test-Path ".venv\Scripts\python.exe")) { Fail "venv creation failed. Is Python 3.11 or newer installed?" }

$venvPy = ".\.venv\Scripts\python.exe"

Write-Host "[2/4] upgrading pip"
& $venvPy -m pip install --quiet --upgrade pip

Write-Host "[3/4] installing core requirements"
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements.txt failed" }

Write-Host "[4/4] installing the ultralytics backend (YOLO + BoT-SORT)"
& $venvPy -m pip install "ultralytics>=8.2,<9"
if ($LASTEXITCODE -ne 0) { Fail "ultralytics install failed" }

Write-Host "`n[check] importing everything"
& $venvPy -c "import numpy,cv2,yaml,onnxruntime,torch,ultralytics;print('numpy',numpy.__version__);print('cv2',cv2.__version__);print('torch',torch.__version__);print('ultralytics',ultralytics.__version__);print('ALL OK')"
if ($LASTEXITCODE -ne 0) { Fail "import check failed, see the output above" }

Write-Host "`nDone. To run:" -ForegroundColor Green
Write-Host "    .venv\Scripts\python.exe run.py YOURCLIP.mp4"
Write-Host "or activate the environment first:"
Write-Host "    .\.venv\Scripts\Activate.ps1"
Read-Host "Press Enter to close"
