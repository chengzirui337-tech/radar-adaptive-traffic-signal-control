param(
    [ValidateSet('view', 'all', 'train', 'evaluate', 'plot', 'report', 'test', 'optimize')]
    [string]$Task = 'view',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$reportPath = Join-Path $PSScriptRoot 'results\full\实验报告.html'
$optimizedReport = Join-Path $PSScriptRoot 'results\optimized\优化训练报告.html'
if (Test-Path -LiteralPath $optimizedReport) { $reportPath = $optimizedReport }
if ($Task -eq 'view' -and (Test-Path -LiteralPath $reportPath)) {
    Invoke-Item -LiteralPath $reportPath
    exit 0
}
if ($Task -eq 'view') { $Task = 'all' }
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
$taskPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } elseif ($pythonCommand) { $pythonCommand.Source } else { $null }
if (-not $taskPython) {
    throw '请先安装 Python 3.12 并执行 python -m pip install -r requirements.txt。'
}
& $taskPython -c 'import numpy, pandas, matplotlib, torch' 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Python 依赖不完整，请执行 python -m pip install -r requirements.txt。'
}
if ($Task -eq 'test') {
    & $taskPython -m unittest discover -s tests -v
} elseif ($Task -eq 'all') {
    & $taskPython -u run.py all --episodes 40 --ablation @ExtraArgs
} elseif ($Task -eq 'evaluate') {
    & $taskPython -u run.py evaluate --ablation @ExtraArgs
} elseif ($Task -eq 'optimize') {
    & $taskPython -u optimize.py longer --episodes 160
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython -u optimize.py ppo --episodes 100 --output results/optimized/ppo
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython -u optimize.py calibrate
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython -u optimize.py ppo4 --episodes 60 --output results/optimized/ppo4
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython -u optimize.py finalize
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $taskPython verify_optimized.py
} else {
    & $taskPython -u run.py $Task @ExtraArgs
}
exit $LASTEXITCODE
