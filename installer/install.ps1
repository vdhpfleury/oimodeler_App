# One-command installer for OIModeler App on Windows.
#
# Usage (from an empty folder -- downloads the app for you):
#   irm https://raw.githubusercontent.com/vdhpfleury/oimodeler_App/main/installer/install.ps1 | iex
#
# If PowerShell blocks the command above (execution policy), download the
# file and run it directly instead:
#   powershell -ExecutionPolicy Bypass -File installer\install.ps1
#
# Usage (from an existing checkout -- reuses it, no download):
#   powershell -ExecutionPolicy Bypass -File installer\install.ps1
#
# What it does: locates a supported Python (3.11-3.13), downloads the app
# if it isn't already present, creates an isolated virtual environment,
# installs every dependency, runs the doctor.py health check, then launches
# the app. See README.md's Compatibility/Troubleshooting sections for why
# each step exists.

$ErrorActionPreference = "Stop"

$RepoArchiveUrl = "https://github.com/vdhpfleury/oimodeler_App/archive/refs/heads/main.zip"
$SupportedVersions = @("3.11", "3.12", "3.13")
$TotalSteps = 6

function Write-Step($Number, $Message) {
    Write-Host ""
    Write-Host "[$Number/$TotalSteps] $Message"
}

function Write-Ok($Message) {
    Write-Host "      OK  $Message"
}

function Write-Fail($Message) {
    Write-Host "      X   $Message"
}

function Find-Python {
    foreach ($v in $SupportedVersions) {
        try {
            $null = & py "-$v" -c "1" 2>$null
            if ($LASTEXITCODE -eq 0) { return @("py", "-$v") }
        } catch {}
    }
    foreach ($cmd in @("python", "python3")) {
        $exists = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($exists) {
            $ver = & $cmd -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($SupportedVersions -contains $ver) { return @($cmd) }
        }
    }
    return $null
}

Write-Host "========================================"
Write-Host "        OIModeler App Installer"
Write-Host "========================================"

Write-Step 1 "Checking prerequisites (Python, Git)..."
$PythonCmd = Find-Python
if ($null -eq $PythonCmd) {
    Write-Fail "No Python 3.11, 3.12 or 3.13 was found on your PATH."
    Write-Host ""
    Write-Host "      OIModeler App needs one of these versions specifically -- see"
    Write-Host "      the Compatibility section in README.md for why. Install one"
    Write-Host "      from https://www.python.org/downloads/ (check 'Add python.exe"
    Write-Host "      to PATH' during setup), then run this installer again."
    exit 1
}
$PyVersion = & $PythonCmd[0] $PythonCmd[1..($PythonCmd.Length - 1)] -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
Write-Ok "Python $PyVersion ($($PythonCmd -join ' '))"

if (Get-Command git -ErrorAction SilentlyContinue) {
    Write-Ok "Git found"
} else {
    Write-Fail "Git was not found on your PATH."
    Write-Host ""
    Write-Host "      pip needs Git to install the oimodeler library from its GitHub"
    Write-Host "      repository (a dependency of this app). Install 'Git for Windows'"
    Write-Host "      from https://git-scm.com/book/en/v2/Getting-Started-Installing-Git"
    Write-Host "      (it adds Git to your PATH automatically), then run this installer"
    Write-Host "      again."
    exit 1
}

Write-Step 2 "Getting the application..."
if ((Test-Path "app.py") -and (Test-Path "requirements.txt")) {
    $AppDir = (Get-Location).Path
    Write-Ok "Already in an OIModeler App checkout ($AppDir)"
} else {
    $AppDir = Join-Path (Get-Location).Path "oimodeler_App"
    if (Test-Path (Join-Path $AppDir "app.py")) {
        Write-Ok "Found an existing download at $AppDir"
    } else {
        try {
            $ZipPath = Join-Path $env:TEMP "oimodeler_App.zip"
            $ExtractPath = Join-Path $env:TEMP "oimodeler_App_extract"
            Invoke-WebRequest -Uri $RepoArchiveUrl -OutFile $ZipPath
            if (Test-Path $ExtractPath) { Remove-Item $ExtractPath -Recurse -Force }
            Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath
            $ExtractedRoot = Get-ChildItem $ExtractPath | Select-Object -First 1
            New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
            Move-Item -Path (Join-Path $ExtractedRoot.FullName "*") -Destination $AppDir -Force
            Remove-Item $ZipPath, $ExtractPath -Recurse -Force
            Write-Ok "Downloaded to $AppDir"
        } catch {
            Write-Fail "Could not download the application from GitHub."
            Write-Fail "Check your internet connection and try again."
            exit 1
        }
    }
}
Set-Location $AppDir

Write-Step 3 "Creating an isolated environment..."
if (-not (Test-Path "env_oim")) {
    & $PythonCmd[0] $PythonCmd[1..($PythonCmd.Length - 1)] -m venv env_oim
}
& ".\env_oim\Scripts\Activate.ps1"
Write-Ok "Environment ready ($AppDir\env_oim)"

Write-Step 4 "Installing dependencies (this can take a few minutes)..."
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Dependency installation failed -- see the error above."
    Write-Fail "For common causes (wrong Python version, missing Git), see the"
    Write-Fail "Troubleshooting section in README.md."
    exit 1
}
Write-Ok "Dependencies installed"

Write-Step 5 "Verifying the installation..."
python doctor.py
if ($LASTEXITCODE -ne 0) {
    Write-Fail "The health check found a problem -- see the report above, and"
    Write-Fail "the Troubleshooting section in README.md."
    exit 1
}
Write-Ok "Installation verified"

Write-Step 6 "Starting OIModeler App..."
Write-Host ""
Write-Host "========================================"
Write-Host "Installation successful! Launching now."
Write-Host "Press Ctrl+C to stop the app."
Write-Host "========================================"
Write-Host ""
streamlit run app.py
