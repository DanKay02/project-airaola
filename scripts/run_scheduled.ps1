param(
    [switch]$SkipEmail
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

$PythonPath = Join-Path `
    $ProjectRoot `
    ".venv\Scripts\python.exe"

$MainPath = Join-Path `
    $ProjectRoot `
    "main.py"

$LogDirectory = Join-Path `
    $ProjectRoot `
    "data\logs"

$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$LogPath = Join-Path `
    $LogDirectory `
    "scheduled_run_$Timestamp.log"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDirectory |
    Out-Null

function Write-RunLog {
    param(
        [Parameter(Mandatory)]
        [string]$Message
    )

    $LogMessage = (
        "[{0}] {1}" -f `
        (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), `
        $Message
    )

    Write-Host $LogMessage

    Add-Content `
        -Path $LogPath `
        -Value $LogMessage `
        -Encoding UTF8
}

Write-RunLog `
    "Project Airaola scheduled run starting."

Write-RunLog `
    "Project root: $ProjectRoot"

Write-RunLog `
    "Log file: $LogPath"

if (-not (Test-Path $PythonPath)) {
    Write-RunLog `
        "FAILED: Virtual-environment Python was not found at $PythonPath"

    exit 1
}

if (-not (Test-Path $MainPath)) {
    Write-RunLog `
        "FAILED: main.py was not found at $MainPath"

    exit 1
}

$ArgumentList = @(
    "`"$MainPath`""
    "--dry-run"
)

if (-not $SkipEmail) {
    $ArgumentList += "--send-email"
}

Write-RunLog `
    "Launching Airaola in protected dry-run mode."

if ($SkipEmail) {
    Write-RunLog `
        "Email delivery is disabled for this run."
}
else {
    Write-RunLog `
        "Email delivery is enabled."
}

$ProcessStartInfo = New-Object `
    System.Diagnostics.ProcessStartInfo

$ProcessStartInfo.FileName = $PythonPath

$ProcessStartInfo.Arguments = (
    $ArgumentList -join " "
)

$ProcessStartInfo.WorkingDirectory = $ProjectRoot
$ProcessStartInfo.UseShellExecute = $false
$ProcessStartInfo.RedirectStandardOutput = $true
$ProcessStartInfo.RedirectStandardError = $true
$ProcessStartInfo.CreateNoWindow = $true

$Process = New-Object `
    System.Diagnostics.Process

$Process.StartInfo = $ProcessStartInfo

try {
    $Started = $Process.Start()

    if (-not $Started) {
        throw "Python process could not be started."
    }

    $StandardOutput = (
        $Process.StandardOutput.ReadToEnd()
    )

    $StandardError = (
        $Process.StandardError.ReadToEnd()
    )

    $Process.WaitForExit()

    if ($StandardOutput) {
        Write-Host $StandardOutput

        Add-Content `
            -Path $LogPath `
            -Value $StandardOutput `
            -Encoding UTF8
    }

    if ($StandardError) {
        Write-Host $StandardError

        Add-Content `
            -Path $LogPath `
            -Value $StandardError `
            -Encoding UTF8
    }

    $ExitCode = $Process.ExitCode
}
catch {
    Write-RunLog `
        "FAILED: $($_.Exception.Message)"

    exit 1
}
finally {
    $Process.Dispose()
}

if ($ExitCode -ne 0) {
    Write-RunLog `
        "Airaola exited with failure code $ExitCode."

    exit $ExitCode
}

Write-RunLog `
    "Project Airaola scheduled run completed successfully."

exit 0