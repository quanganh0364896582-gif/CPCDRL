param([int]$N = 60)

$python = (Get-Command python).Source
$dir    = $PSScriptRoot

# Parse edge_device and cloud_device lists from config.yaml
$yaml = Get-Content "$dir\config.yaml" -Raw

$edgeMatch = [regex]::Match($yaml, 'edge_device\s*:\s*\[([^\]]+)\]')
if ($edgeMatch.Success) {
    $edgeDevices = $edgeMatch.Groups[1].Value -split ',' |
                   ForEach-Object { $_.Trim().Trim('"').Trim("'") } |
                   Where-Object { $_ -ne '' }
} else {
    $edgeMatch2 = [regex]::Match($yaml, 'edge_device\s*:\s*"?([A-Z0-9_]+)"?')
    $edgeDevices = @($edgeMatch2.Groups[1].Value.Trim())
}

$cloudMatch = [regex]::Match($yaml, 'cloud_device\s*:\s*\[([^\]]+)\]')
if ($cloudMatch.Success) {
    $cloudDevices = $cloudMatch.Groups[1].Value -split ',' |
                    ForEach-Object { $_.Trim().Trim('"').Trim("'") } |
                    Where-Object { $_ -ne '' }
} else {
    $cloudMatch2 = [regex]::Match($yaml, 'cloud_device\s*:\s*"?([A-Z0-9_]+)"?')
    $cloudDevices = @($cloudMatch2.Groups[1].Value.Trim())
}

Write-Host "Edge devices : $edgeDevices" -ForegroundColor Cyan
Write-Host "Cloud devices: $cloudDevices" -ForegroundColor Cyan
Write-Host "Running $N episodes. Python: $python" -ForegroundColor Cyan

for ($i = 1; $i -le $N; $i++) {
    Write-Host "`n=== Episode $i / $N ===" -ForegroundColor Cyan

    $srv = Start-Process $python -ArgumentList "server.py" `
           -WorkingDirectory $dir -PassThru -NoNewWindow
    Start-Sleep -Seconds 2

    $procs = @()

    # One edge process per device, each knows its hardware profile
    foreach ($dev in $edgeDevices) {
        $p = Start-Process $python `
             -ArgumentList "client.py","--layer_id","1","--edge_device",$dev `
             -WorkingDirectory $dir -PassThru -NoNewWindow
        $procs += $p
    }

    # One cloud process per cloud device
    $cloudLayerId = 2
    foreach ($dev in $cloudDevices) {
        $p = Start-Process $python `
             -ArgumentList "client.py","--layer_id",$cloudLayerId `
             -WorkingDirectory $dir -PassThru -NoNewWindow
        $procs += $p
        $cloudLayerId++
    }

    $srv.WaitForExit()
    foreach ($p in $procs) { $p.WaitForExit() }

    Write-Host "Episode $i done." -ForegroundColor Green
}

Write-Host "`nAll $N episodes finished." -ForegroundColor Green
