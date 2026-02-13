param(
    [string]$PythonExe = "python.exe"
)

$ErrorActionPreference = "Stop"

$configs = @(
    "configs/cdgp_week1_toy_cifar10_random.json",
    "configs/cdgp_week1_toy_cifar10_entropy.json",
    "configs/cdgp_week1_toy_cifar10_domain_guided.json"
)

Write-Host "Python: $PythonExe"
& $PythonExe -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

foreach ($config in $configs) {
    Write-Host ""
    Write-Host "=== Running $config ==="
    & $PythonExe src/main.py --mode phase1b --config $config
}

Write-Host ""
Write-Host "Week 1 toy runs completed."
