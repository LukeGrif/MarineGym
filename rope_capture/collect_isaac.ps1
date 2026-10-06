# Several Isaac Sim capture runs in one go, on Windows (see collect_isaac.sh).
#   .\collect_isaac.ps1 C:\rope_isaac                      # 5 runs x 2000, 1920x1080
#   .\collect_isaac.ps1 C:\rope_isaac -Runs 2 -Count 500 -Size 960x540 -Worlds mixed,reef
# Run it from the Python environment that has Isaac Sim (python on the PATH).
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Runs = 5,
    [int]$Count = 2000,
    [string]$Size = "1920x1080",
    [string[]]$Worlds = @("mixed")
)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
for ($i = 1; $i -le $Runs; $i++) {
    $Run = Join-Path $Out "isaac_run$i"
    if (Test-Path (Join-Path $Run "done")) { Write-Host "isaac_run$i already done"; continue }
    $World = $Worlds[($i - 1) % $Worlds.Count]
    $Seed = 1000 + $i
    Write-Host "isaac_run${i}: $Count pictures ($Size), seed $Seed, world $World"
    python (Join-Path $Here "capture.py") $Run --count $Count --size $Size --seed $Seed --world $World
    if ($LASTEXITCODE -ne 0) { throw "capture failed for isaac_run$i" }
}
Write-Host "done: copy $Out to the training machine and train with  python3 dataset/train_seg.py <runs...>"
