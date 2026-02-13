param(
    [string]$PythonExe = "python.exe",
    [string]$ConfigDir = "configs/week1_batch",
    [switch]$IncludeBaselines
)

$ErrorActionPreference = "Stop"

$configs = @(
    "$ConfigDir/cdgp_week1_batch_cifar10_domain_guided_calibrated_w02.json",
    "$ConfigDir/cdgp_week1_batch_cifar10_domain_guided_calibrated_w05.json",
    "$ConfigDir/cdgp_week1_batch_cifar10_domain_guided_calibrated_w08.json"
)

if ($IncludeBaselines) {
    $configs = @(
        "$ConfigDir/cdgp_week1_batch_cifar10_random.json",
        "$ConfigDir/cdgp_week1_batch_cifar10_entropy.json"
    ) + $configs
}

Write-Host "Python: $PythonExe"
& $PythonExe -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

foreach ($config in $configs) {
    if (-not (Test-Path $config)) {
        throw "Missing config: $config"
    }
    Write-Host ""
    Write-Host "=== Running $config ==="
    & $PythonExe src/main.py --mode phase1b --config $config
}

Write-Host ""
Write-Host "Week 1 calibrated batch runs completed."
