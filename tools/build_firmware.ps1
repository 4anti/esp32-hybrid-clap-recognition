param(
    [ValidateSet('shadow', 'active', 'dsp')][string]$Mode = 'shadow',
    [string]$Port = 'COM3',
    [switch]$Upload
)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$cliCommand = Get-Command arduino-cli -ErrorAction SilentlyContinue
$cliPath = if ($cliCommand) { $cliCommand.Source } else {
    Join-Path $env:LOCALAPPDATA 'Programs\Arduino IDE\resources\app\lib\backend\resources\arduino-cli.exe'
}
if (!(Test-Path -LiteralPath $cliPath)) {
    throw 'Arduino CLI was not found. Install Arduino IDE and Arduino-ESP32 3.3.11.'
}
$buildName = if ($Mode -ne 'dsp') { 'ai' } else { 'dsp' }
$outputName = if ($Mode -ne 'dsp') { "ai-$Mode" } else { 'dsp' }
$buildPath = Join-Path $projectPath ".build\compile-$buildName"
$outputPath = Join-Path $projectPath ".build\$outputName"
$enabled = if ($Mode -ne 'dsp') { 1 } else { 0 }
$shadow = if ($Mode -eq 'active') { 0 } else { 1 }
$board = 'esp32:esp32:esp32'
& $cliPath compile --fqbn $board --jobs 4 --build-property "compiler.cpp.extra_flags=-DCLAP_ENABLE_AI=$enabled -DCLAP_AI_SHADOW=$shadow" --build-path $buildPath --output-dir $outputPath (Join-Path $projectPath 'clap_double')
if ($LASTEXITCODE -ne 0) { throw 'Firmware compilation failed; nothing was uploaded.' }
if ($Upload) {
    & $cliPath upload --fqbn $board --port $Port --input-dir $outputPath
    if ($LASTEXITCODE -ne 0) { throw 'Firmware upload failed.' }
}
