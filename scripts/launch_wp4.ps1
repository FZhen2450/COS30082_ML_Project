param(
    [Parameter(Mandatory = $true)][string]$Python,
    [string]$ClipPackages = '',
    [string]$Protocol = '',
    [string]$Images = '',
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = (Resolve-Path -LiteralPath $Python).Path
if (-not $Protocol) { $Protocol = Join-Path $projectRoot 'artifacts/data/development_seed42_reconciled_v1.json' }
if (-not $Images) { $Images = Join-Path $projectRoot 'DPD_Small_dataset' }
$protocolPath = (Resolve-Path -LiteralPath $Protocol).Path
$imagePath = (Resolve-Path -LiteralPath $Images).Path
$previousPythonPath = $env:PYTHONPATH
try {
    if ($ClipPackages) {
        $packagePath = (Resolve-Path -LiteralPath $ClipPackages).Path
        if ((Test-Path -LiteralPath (Join-Path $packagePath 'torch')) -or
            (Test-Path -LiteralPath (Join-Path $packagePath 'torchvision'))) {
            throw 'The optional CLIP package folder must not override Torch or Torchvision.'
        }
        $env:PYTHONPATH = $packagePath
        if ($previousPythonPath) { $env:PYTHONPATH += [IO.Path]::PathSeparator + $previousPythonPath }
    }
    if ($CheckOnly) {
        $preflightCode = "import sys,importlib.util,json,torch,torchvision,tkinter; assert sys.version_info >= (3,10); print(json.dumps(dict(python=sys.executable,torch=str(torch.__version__),torchvision=str(torchvision.__version__),clip_available=importlib.util.find_spec('clip') is not None,tk_version=tkinter.TkVersion),indent=2))"
        & $pythonPath -B -c $preflightCode
    } else {
        & $pythonPath -B (Join-Path $projectRoot 'gui.py') --protocol $protocolPath --images $imagePath --source (Join-Path $projectRoot 'DPD_Small_train.csv')
    }
    if ($LASTEXITCODE -ne 0) { throw "LeafLens Python process exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
