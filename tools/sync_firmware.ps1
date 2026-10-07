param([string]$Destination = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'Arduino\clap_double'))
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$sourcePath = Join-Path $projectPath 'clap_double'
$backupPath = Join-Path $projectPath ('.build\arduino-backup-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
New-Item -ItemType Directory -Path $backupPath -Force | Out-Null
$files = Get-ChildItem -LiteralPath $sourcePath -File | Where-Object { $_.Extension -in '.ino', '.h' }
foreach ($file in $files) {
    $targetPath = Join-Path $Destination $file.Name
    if (Test-Path -LiteralPath $targetPath) {
        Copy-Item -LiteralPath $targetPath -Destination (Join-Path $backupPath $file.Name)
    }
    Copy-Item -LiteralPath $file.FullName -Destination $targetPath -Force
    if ((Get-FileHash -LiteralPath $file.FullName).Hash -ne (Get-FileHash -LiteralPath $targetPath).Hash) {
        throw "Firmware copy verification failed: $($file.Name)"
    }
}
Write-Host "Synced $($files.Count) firmware files to $Destination. Previous files are saved in $backupPath."
