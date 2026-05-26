param(
    [int]$N = 200,
    [string[]]$Devices = @("DEVICE_4", "DEVICE_7")
)

$dir    = $PSScriptRoot
$config = "$dir\config.yaml"

function Set-ConfigField($yaml, $key, $value) {
    $yaml -replace "(?m)^(\s*$key\s*:\s*).*$", "`${1}$value"
}

foreach ($device in $Devices) {
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  Training $device  ($N episodes)" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow

    # Update config.yaml: max_frames=160, edge_device
    $yaml = Get-Content $config -Raw
    $yaml = Set-ConfigField $yaml "max_frames" '160  # limit frames per episode for faster RL training (0 = no limit)'
    $yaml = $yaml -replace 'edge_device\s*:\s*\[[^\]]+\]', "edge_device:  [""$device""]"
    $yaml | Set-Content $config -Encoding utf8 -NoNewline

    Write-Host "Config updated: max_frames=160, edge_device=$device" -ForegroundColor Cyan

    # Run loop
    & "$dir\run_loop.ps1" -N $N

    Write-Host "`n[OK] $device training complete." -ForegroundColor Green
}

Write-Host "`nAll devices trained." -ForegroundColor Green
