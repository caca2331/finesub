function Invoke-NativeCommand {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,
        [Parameter(Mandatory = $true)]
        [string]$StdoutPath,
        [Parameter(Mandatory = $true)]
        [string]$StderrPath,
        [switch]$RedactSensitiveEnvironment
    )

    $PreviousErrorActionPreference = $ErrorActionPreference
    $SensitiveVariables = @()
    try {
        if ($RedactSensitiveEnvironment) {
            $ProcessEnvironment = [System.Environment]::GetEnvironmentVariables(
                [System.EnvironmentVariableTarget]::Process
            )
            $SensitiveVariables = @(
                foreach ($Name in $ProcessEnvironment.Keys) {
                    if ($Name -match "(API[_-]?KEY|TOKEN|SECRET|PASSWORD|PRIVATE[_-]?KEY|CREDENTIAL)") {
                        [PSCustomObject]@{
                            Name = [string]$Name
                            Value = [string]$ProcessEnvironment[$Name]
                        }
                    }
                }
            )
            foreach ($Variable in $SensitiveVariables) {
                [System.Environment]::SetEnvironmentVariable(
                    $Variable.Name,
                    $null,
                    [System.EnvironmentVariableTarget]::Process
                )
            }
        }

        # Windows PowerShell 5 converts native stderr into ErrorRecord objects.
        # With the caller's Stop preference, normal compiler progress becomes a
        # terminating NativeCommandError even when the process exits with zero.
        $ErrorActionPreference = "Continue"
        & $FilePath @ArgumentList 1> $StdoutPath 2> $StderrPath
        $ExitCode = $LASTEXITCODE
    }
    finally {
        foreach ($Variable in $SensitiveVariables) {
            [System.Environment]::SetEnvironmentVariable(
                $Variable.Name,
                $Variable.Value,
                [System.EnvironmentVariableTarget]::Process
            )
        }
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    return [int]$ExitCode
}
