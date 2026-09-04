$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$previousElanHome = $env:ELAN_HOME
$locationDepth = 0
$runtimePython = if ($env:WHISPER_RUNTIME_PYTHON) {
    $env:WHISPER_RUNTIME_PYTHON
} else {
    "python"
}

try {
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    Push-Location $projectRoot
    $locationDepth += 1
    & $runtimePython -B -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw "Runtime tests failed with exit code $LASTEXITCODE." }
    & $runtimePython -B -m unittest discover -s tools -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Tool tests failed with exit code $LASTEXITCODE." }
    & $runtimePython -m ruff check src tests tools examples infra
    if ($LASTEXITCODE -ne 0) { throw "Ruff checks failed with exit code $LASTEXITCODE." }
    & $runtimePython -m ruff format --check src tests tools examples infra
    if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed with exit code $LASTEXITCODE." }
    & $runtimePython -m mypy src
    if ($LASTEXITCODE -ne 0) { throw "Mypy checks failed with exit code $LASTEXITCODE." }
    & $runtimePython -B tools/check_repository.py
    if ($LASTEXITCODE -ne 0) { throw "Repository checks failed with exit code $LASTEXITCODE." }
    & $runtimePython -B examples/minimal_transaction.py
    if ($LASTEXITCODE -ne 0) { throw "Minimal example failed with exit code $LASTEXITCODE." }
    & $runtimePython -B -m compileall -q src tools examples infra
    if ($LASTEXITCODE -ne 0) { throw "Python compilation failed with exit code $LASTEXITCODE." }
    Pop-Location
    $locationDepth -= 1

    $defaultElanHome = Join-Path $env:USERPROFILE ".elan"
    if (-not $env:ELAN_HOME -and (Test-Path $defaultElanHome)) {
        $env:ELAN_HOME = $defaultElanHome
    }
    Push-Location (Join-Path $projectRoot "formal\lean")
    $locationDepth += 1
    $toolchainName = (Get-Content "lean-toolchain" -Raw).Trim()
    $toolchainDirectory = $toolchainName.Replace("/", "--").Replace(":", "---")
    $pinnedLake = Join-Path $env:ELAN_HOME "toolchains\$toolchainDirectory\bin\lake.exe"
    if (Test-Path $pinnedLake) {
        & $pinnedLake build
    } else {
        lake build
    }
    if ($LASTEXITCODE -ne 0) { throw "Lean build failed with exit code $LASTEXITCODE." }
    Pop-Location
    $locationDepth -= 1
}
finally {
    while ($locationDepth -gt 0) {
        Pop-Location
        $locationDepth -= 1
    }
    $env:PYTHONPATH = $previousPythonPath
    $env:ELAN_HOME = $previousElanHome
}
