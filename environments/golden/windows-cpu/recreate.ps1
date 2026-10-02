# 保存したWindows環境を新規venvへ再構築する。既存環境は上書きしない。
param(
    [string]$PythonExe = "py",
    [string]$VenvPath = "venv/golden-rebuild"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot/../../..").Path
$reference = Get-Content -LiteralPath "$PSScriptRoot/environment.json" -Raw -Encoding UTF8 | ConvertFrom-Json
if ($env:OS -ne "Windows_NT") { throw "この環境定義はWindows専用です。" }
$pythonArgs = @()
if ($PythonExe -eq "py") { $pythonArgs = @("-3.13") }
$version = & $PythonExe @pythonArgs -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0) { throw "Pythonを起動できません。" }
if ($version.Trim() -ne $reference.python.version) {
    throw "Python $($reference.python.version)が必要です。現在: $version"
}
$target = [IO.Path]::GetFullPath((Join-Path $repoRoot $VenvPath))
if (Test-Path -LiteralPath $target) { throw "保存先が存在します。新しいVenvPathを指定してください: $target" }
& $PythonExe @pythonArgs -m venv $target
if ($LASTEXITCODE -ne 0) { throw "venv作成に失敗しました。" }
$targetPython = Join-Path $target "Scripts/python.exe"
# PyPIのWindows版torchは配布版2.12.1、実行時表示2.12.1+cpu。別CPU wheelへ置換しない。
& $targetPython -m pip --isolated install --index-url $reference.package_index -r "$PSScriptRoot/requirements-freeze.txt"
if ($LASTEXITCODE -ne 0) { throw "依存パッケージのインストールに失敗しました。" }
& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "依存の整合性検査に失敗しました。" }
Write-Output "Created: $targetPython"
