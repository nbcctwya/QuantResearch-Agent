"""WSL2 host sleep inhibitor, scoped to the active research study's lifetime."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from . import ARTIFACTS
from .common import now,write_json


def launch():
    distribution = os.environ.get("WSL_DISTRO_NAME")
    executable = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
    if not distribution or not executable.exists():
        return None
    state_path = ARTIFACTS/"study/state.json"
    windows_path = str(state_path).replace("/","\\")
    unc = "\\\\wsl.localhost\\"+distribution+windows_path
    literal_path = unc.replace("'","''")
    script = r'''
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ResearchPower {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$previous = [ResearchPower]::SetThreadExecutionState([uint32]2147483649)
if ($previous -eq 0) { throw 'SetThreadExecutionState failed' }
Write-Output 'Research sleep inhibitor acquired: SYSTEM_REQUIRED; display may sleep.'
$maximum = [DateTime]::UtcNow.AddHours(120)
try {
    while ([DateTime]::UtcNow -lt $maximum) {
        $state = Get-Content -Raw -LiteralPath '__STATE_PATH__' | ConvertFrom-Json
        if ($state.phase -eq 'campaign_finished' -or $state.phase -eq 'stopped_by_control') { break }
        $heartbeat = [DateTimeOffset]::Parse($state.heartbeat).UtcDateTime
        if (([DateTime]::UtcNow - $heartbeat).TotalMinutes -gt 15) { break }
        Start-Sleep -Seconds 20
    }
} finally {
    [void][ResearchPower]::SetThreadExecutionState([uint32]2147483648)
    Write-Output 'Research sleep inhibitor released.'
}
'''.replace("__STATE_PATH__",literal_path)
    folder = ARTIFACTS/"study"
    folder.mkdir(parents=True,exist_ok=True)
    metadata = folder/"keep_awake.json"
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        command = Path(f"/proc/{previous['pid']}/cmdline")
        if command.exists() and b"powershell.exe" in command.read_bytes():
            return previous["pid"]
    (folder/"keep_awake.ps1").write_text(script)
    with (folder/"keep_awake.log").open("a") as log:
        process = subprocess.Popen([str(executable),"-NoProfile","-NonInteractive","-Command",script],
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True,close_fds=True)
    write_json(metadata,{"pid":process.pid,"started_at":now(),"state_path":str(state_path),
                         "scope":"temporary thread execution request; no persistent power-plan changes"})
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"Sleep inhibitor exited: {(folder/'keep_awake.log').read_text()[-2000:]}")
    return process.pid


if __name__ == "__main__":
    print("Sleep inhibitor PID:",launch(),flush=True)
