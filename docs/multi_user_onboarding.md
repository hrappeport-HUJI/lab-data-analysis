# Shared Windows lab PC onboarding

## Prerequisites & Architecture Overview

The lab PC has a 10-core/20-thread Intel Core i9-9900X, 128 GB RAM, and one NVIDIA RTX 2080 Ti with 11 GB VRAM. You need a lab administrator, your own T3 and GitHub accounts, and your own login for at least one AI provider. The PC must stay powered on and awake for remote access. Ask the repository maintainer for a published revision containing the integrated v2.2 pipeline, GPU lock and Position Viewer v2.

The owner keeps the Windows account `t3`, Scheduled Task `T3 Code Server`, and `C:\T3\run-t3.ps1` as configured. Each additional lab member gets a **Standard Windows account** (`t3-<name>`), a separate password-backed Scheduled Task and loopback port, and a T3 Connect tunnel paired to their own T3 account. Windows keeps the profiles, credentials and workspaces separate. The outbound Connect tunnel reaches the member's T3 server without opening its port to the LAN. T3, GitHub and provider sign-ins are separate. [T3 Connect setup](https://github.com/pingdotgg/t3code/blob/main/docs/user/remote-access.md)

| Resource | Where it lives |
| --- | --- |
| Shared read-only Python and models | `E:\microscopy\python`, `E:\microscopy\env-gpu` (CUDA/PyTorch/Omnipose), `E:\microscopy\env` (CPU/analysis), and `E:\microscopy\models` |
| Personal workspace | `E:\microscopy\users\<name>\` or the assigned directory on the multi-TB SSD |
| Personal workspace contents | `lab-data-analysis\` (clone), `data\` (acquisitions), `work\` (projects, results and caches), and `logs\` |
| Ports | One T3 port, e.g. `3774`, and one disjoint viewer range, e.g. `8900–8909`. The next member could use `3775` and `8910–8919`. |

## Part 1 — Admin Setup on the Lab PC (once per new person, ~5–10 min)

1. Reserve ports in the lab's allocation list. Run the blocks below in the same **elevated 64-bit Windows PowerShell** session. Edit the first four values for the member. For the dedicated SSD, set `$workspace` to an unused directory on that NTFS volume.

   ```powershell
   $name = 'maya'
   $t3Port = 3774
   $viewerPorts = 8900..8909
   $githubUser = 'REPLACE_WITH_GITHUB_LOGIN'

   if ($name -notmatch '^[a-z][a-z0-9-]{0,16}$') { throw 'Use a short lowercase account name.' }
   $account = "t3-$name"
   $principal = "$env:COMPUTERNAME\$account"
   $workspace = "E:\microscopy\users\$name"
   $script = "C:\T3\run-t3-$name.ps1"
   $task = "T3 Code Server - $name"
   if ((Get-LocalUser -Name $account -ErrorAction SilentlyContinue) -or
       (Test-Path -LiteralPath $workspace) -or (Test-Path -LiteralPath $script) -or
       (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue)) {
       throw 'An account, workspace, script or task already exists. Check the allocation.'
   }
   $busy = Get-NetTCPConnection -State Listen |
       Where-Object { $_.LocalPort -in (@($t3Port) + $viewerPorts) }
   if ($busy -or $t3Port -in $viewerPorts) { throw 'Choose another port allocation.' }

   $password = Read-Host "Windows password for $principal" -AsSecureString
   $member = New-LocalUser -Name $account -Password $password -Description 'Lab T3 Code member' -ErrorAction Stop
   $usersGroup = Get-LocalGroup -SID 'S-1-5-32-545'
   if ($member.SID.Value -notin (Get-LocalGroupMember -Group $usersGroup).SID.Value) {
       Add-LocalGroupMember -Group $usersGroup -Member $member -ErrorAction Stop
   }
   $memberSid = '*' + $member.SID.Value

   function Set-LabAcl {
       & icacls.exe @args | Out-Host
       if ($LASTEXITCODE -ne 0) { throw 'icacls failed. Check the path and ACL before proceeding.' }
   }
   New-Item -ItemType Directory -Path $workspace -Force -ErrorAction Stop | Out-Null
   Set-LabAcl $workspace /inheritance:r /grant:r '*S-1-5-18:(OI)(CI)F' '*S-1-5-32-544:(OI)(CI)F' "${memberSid}:(OI)(CI)M"
   foreach ($folder in 'data', 'work', 'logs') {
       New-Item -ItemType Directory -Path (Join-Path $workspace $folder) -ErrorAction Stop | Out-Null
   }
   foreach ($folder in 'python', 'env-gpu', 'env', 'models') {
       Set-LabAcl "E:\microscopy\$folder" /grant:r "${memberSid}:(OI)(CI)RX"
   }
   ```

   The workspace ACL grants Modify to the member and Full Control to Administrators and SYSTEM. Check effective access on the shared directories, including inherited group permissions and any protected descendants. The member needs read/execute on the shared Python and models and write access to their own workspace. Grant access to a separate raw archive explicitly for the authorized acquisition.

2. Check for `E:\microscopy\models\bact_phase_omnitorch_0`, the weight for the GPU environment's `bact_phase_omni` model. If it is missing, the administrator can stage a verified copy from the existing cache at `C:\Users\t3\.cellpose\models\bact_phase_omnitorch_0` without granting access to the owner's profile. Use the shared model directory in all new accounts.

3. Create the startup script and password-backed task. The script uses the member's own T3 CLI, binds to their assigned loopback port, and writes logs in their workspace. Start the task after the member pairs their account in Part 2.

   ```powershell
   $scriptBody = @'
   $env:Path = "$env:USERPROFILE\.local\bin;$env:APPDATA\npm;C:\Program Files\nodejs;C:\Program Files\Git\cmd;" + $env:Path
   $env:T3CODE_PORT = '__PORT__'
   $root = '__WORKSPACE__'
   $log = Join-Path $root 'logs\t3-serve.log'
   Set-Location -LiteralPath $root
   while ($true) {
       if ((Test-Path -LiteralPath $log) -and ((Get-Item -LiteralPath $log).Length -gt 20MB)) {
           Move-Item -LiteralPath $log -Destination "$log.old" -Force
       }
       $cli = Join-Path $env:APPDATA 'npm\t3.cmd'
       if (Test-Path -LiteralPath $cli) {
           Add-Content -LiteralPath $log -Encoding UTF8 "[$(Get-Date -Format o)] Starting T3 Code on port $env:T3CODE_PORT"
           & $cli serve --no-browser --host 127.0.0.1 --port $env:T3CODE_PORT 2>&1 |
               Out-File -LiteralPath $log -Append -Encoding UTF8
           Add-Content -LiteralPath $log -Encoding UTF8 "[$(Get-Date -Format o)] T3 Code exited with $LASTEXITCODE"
       } else {
           Add-Content -LiteralPath $log -Encoding UTF8 "[$(Get-Date -Format o)] Waiting for $cli"
       }
       Start-Sleep -Seconds 15
   }
   '@
   $scriptBody.Replace('__PORT__', "$t3Port").Replace('__WORKSPACE__', $workspace.Replace("'", "''")) |
       Set-Content -LiteralPath $script -Encoding UTF8 -ErrorAction Stop
   Set-LabAcl $script /inheritance:r /grant:r '*S-1-5-18:F' '*S-1-5-32-544:F' "${memberSid}:RX"
   $powershell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
   $arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $script
   $action = New-ScheduledTaskAction -Execute $powershell -Argument $arguments -WorkingDirectory $workspace
   $trigger = New-ScheduledTaskTrigger -AtStartup
   $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
   $registration = @{
       TaskName = $task
       Action = $action
       Trigger = $trigger
       Settings = $settings
       User = $principal
       RunLevel = 'Limited'
   }
   try {
       Register-ScheduledTask @registration -Password ([pscredential]::new($principal, $password).GetNetworkCredential().Password) -ErrorAction Stop | Out-Null
   } finally {
       Remove-Variable password
   }
   (Get-ScheduledTask -TaskName $task).Principal | Select-Object UserId, LogonType, RunLevel
   ```

   Verify `LogonType` is `Password` and `RunLevel` is `Limited`. A password-backed task can run before the member signs in. The administrator must update the task's stored credential when the Windows password changes. Keep passwords and pairing codes out of logs and agent prompts.

4. Invite the member through the GitHub repository's **Settings → Collaborators** or **Collaborators and teams**. A repository admin already signed in with `gh` can send the invitation below. Choose the appropriate access level if the repository is organization-owned. The member must accept the invitation with their own GitHub account. [GitHub collaborator API](https://docs.github.com/en/rest/collaborators/collaborators#add-a-repository-collaborator)

   ```powershell
   gh api --method PUT "repos/hrappeport-HUJI/lab-data-analysis/collaborators/$githubUser"
   ```

5. Give the member their Windows account, workspace path, T3 port, viewer range, and the published v2.2 revision. Confirm that all GPU users can access the same integrated default GPU lock. If the lab sets `LAB_MICROSCOPY_GPU_LOCK`, provision one common path with the access required by the lock implementation. Coordinate the same setting for any existing account that uses the GPU.

## Part 2 — New User First-Time Setup (~5 min)

1. Sign in to Windows once as `t3-<name>` so Windows creates your profile. Alternatively, from the lab PC use `runas.exe /profile /user:".\t3-maya" powershell.exe` with your assigned name. Run the following in that user's **non-elevated** PowerShell. Use the lab-approved T3 version. The example matches the CLI installed on the lab PC when this guide was written.

   ```powershell
   whoami
   $workspace = 'E:\microscopy\users\maya'
   $t3Port = 3774
   $env:Path = "$env:USERPROFILE\.local\bin;$env:APPDATA\npm;C:\Program Files\nodejs;C:\Program Files\Git\cmd;" + $env:Path
   npm.cmd install --global --prefix "$env:APPDATA\npm" t3@0.1.8
   t3.cmd --version
   ```

2. Pair with your own T3 account using the assigned port. `t3 up` can create its own per-user service during pairing. After pairing, use the lab's password-backed task as the sole launcher for your profile.

   ```powershell
   $env:T3CODE_PORT = "$t3Port"
   $env:T3CODE_HOST = '127.0.0.1'
   $t3Home = Join-Path $env:USERPROFILE '.t3'
   $serviceEnv = Join-Path $t3Home 'service.env'
   New-Item -ItemType Directory -Path $t3Home -Force | Out-Null
   if (Test-Path -LiteralPath $serviceEnv) { throw 'Existing T3 configuration. Ask the administrator to check this profile.' }
   @("T3CODE_PORT=$t3Port", 'T3CODE_HOST=127.0.0.1') |
       Set-Content -LiteralPath $serviceEnv -Encoding Ascii
   t3.cmd up --connect --no-lan
   t3.cmd status
   ```

   Finish the browser sign-in and check that your environment is paired. If `t3.cmd service status` shows that `up` installed a service for **your** account, run `t3.cmd service uninstall` from that same profile before the handoff. This preserves the T3 data and Connect pairing. Ask the administrator to run the following in the admin session from Part 1.

   ```powershell
   Start-ScheduledTask -TaskName $task
   ```

   Sign in to T3 Code on your own device with the same T3 account and open your connected environment. Check the server log in your workspace if it stays offline. [T3 background service](https://github.com/pingdotgg/t3code/blob/main/docs/user/background-service.md) · [T3 Connect troubleshooting](https://github.com/pingdotgg/t3code/blob/main/docs/user/t3-connect-troubleshooting.md)

3. Install your chosen provider CLI per account if needed. In T3 Code, open **Settings → Providers** for the connected environment and use your own subscription or approved billing account. For host-side CLI sign-in from the `t3-<name>` profile, use `claude auth login` for Claude Code, `codex login` for Codex, or launch `gemini` and select Google sign-in if Gemini is available in your T3 release. Confirm the provider shows as available and send a short test request. [T3 provider setup](https://github.com/pingdotgg/t3code/blob/main/docs/user/install.md)

4. Accept the GitHub invitation and clone from the same Windows profile. Replace the identity below. For a v2.2 revision outside the default branch, use `git switch --detach REVISION` with the supplied commit, tag or `origin/BRANCH`. Add the clone as a project in your **connected lab PC environment** in T3 Code. Give your agent the separate absolute acquisition and output paths under your workspace.

   ```powershell
   git --version
   gh auth login --hostname github.com --git-protocol https --web
   gh auth setup-git
   git config --global user.name 'Your Name'
   git config --global user.email 'YOUR_GITHUB_EMAIL'
   $repo = Join-Path $workspace 'lab-data-analysis'
   git clone https://github.com/hrappeport-HUJI/lab-data-analysis.git $repo
   Set-Location -LiteralPath $repo
   git status --short --branch
   git rev-parse HEAD
   ```

5. Set up each PowerShell process that runs microscopy. Put your clone on `PYTHONPATH` and use the shared interpreters. The `-B` flag suppresses bytecode writes into shared directories. The module entry point `python -m microscopy` is equivalent to `lab-microscopy`. This avoids duplicating a roughly 15 GB environment.

   ```powershell
   $repo = 'E:\microscopy\users\maya\lab-data-analysis'
   Set-Location -LiteralPath $repo
   $env:PYTHONPATH = $repo
   $env:CELLPOSE_LOCAL_MODELS_PATH = 'E:\microscopy\models'
   $cpu = 'E:\microscopy\env\Scripts\python.exe'
   $gpu = 'E:\microscopy\env-gpu\Scripts\python.exe'

   & $cpu -B -c "import microscopy; print(microscopy.__file__)"
   & $cpu -B -m microscopy mm-metadata --help
   & $cpu -B -m microscopy.position_viewer build --help
   & $gpu -B -c "import importlib.util; assert importlib.util.find_spec('microscopy.gpu_lock'), 'Missing integrated GPU lock'"
   & $gpu -B -m microscopy.fast_omnipose --help
   ```

   Check that the import path is inside your clone. If the viewer or lock is missing, check the revision with the maintainer before starting GPU analysis. Copy these settings into each agent terminal or your own launcher. Use the default shared lock unless the administrator gave everyone a common override. Keep `LAB_MICROSCOPY_DISABLE_GPU_LOCK` unset.

## Part 3 — Day-to-Day Agent Workflow

Start a thread in your lab PC project. Fill in the paths and assigned viewer ports.

```text
Use AGENTS.md, skills/microscopy-experiment/SKILL.md and its workflow
reference. My clone is E:\microscopy\users\maya\lab-data-analysis.
The raw acquisition is E:\microscopy\users\maya\data\EXPERIMENT.
Write configs, caches, logs and outputs under a new directory in
E:\microscopy\users\maya\work. Treat the acquisition as read-only.
Use Position Viewer port 8900 and Review Workbench port 8901.

Use the shared CPU/GPU interpreters and read-only model directory from
docs/multi_user_onboarding.md. Check the checkout revision, GPU lock,
free disk and available RAM. Inspect raw TIFFs, fields, channel mapping
and frame coverage. Extract a Micro-Manager CSV with python -m microscopy
mm-metadata where applicable. Check elapsed times, exposures and channel
IDs. Ask me about missing acquisition details or analysis decisions.

Run a bounded GPU pilot with microscopy.fast_omnipose and its online
founder tracker. Keep the shared GPU lock enabled. Save candidate JSON
and labels in the pilot folder. Report queued time separately. Once
inference starts, verify progress and arm a terminal-state and stall
watcher as described in AGENTS.md. Run heavy work below normal priority.
Report singlets, tagged frame-0 doublets, excluded founders and their
reasons, contact/engulfment losses, and PI annotation checks. Leave
ghost_lysis disabled. Show me representative events before scaling up.

Build a Position Viewer v2 cache with inputs supported by the checkout
and use python -m microscopy.position_viewer --cache-dir <cache> --port 8900.
Give me the access URL through the approved local or Tailscale route.
For saved corrections, use a compatible lab-microscopy project with
tracking.events set to founder_v2 and open python -m microscopy.workbench
on port 8901. Use complete segmentation coverage for measurements that
exclude all detected cells from fluorescence backgrounds. Save
corrections through the review API and keep the original run.

After review, analyze the explicit run ID with --reviewed-only. Use
microscopy.founder_plots to generate the requested figures from the
frozen result directory with verified measurement and eligibility
choices. Report the commit, run ID, review revision, excluded and missing
counts, and result path. Check with me before expanding the pilot.
```

For an on-PC browser, use `--host 127.0.0.1 --no-auth` on Position Viewer only when other users on that PC may see the data. Loopback is shared by Windows users. For approved remote review, use key authentication and restrict access to your own client address, for example `--host 0.0.0.0 --allow-client 100.101.102.103/32` with your client's actual Tailscale IP. Have the administrator allow only the assigned viewer port and authorized address through Windows Firewall. Keep access URLs with keys private. The T3 server port and viewer ports are separate. See the [Review Workbench network options](microscopy/README.md#reviewing-from-another-computer).

## Part 4 — Quick Troubleshooting Table

| Symptom | Check and fix |
| --- | --- |
| Scheduled Task or T3 environment offline | The administrator can run `Get-ScheduledTaskInfo -TaskName 'T3 Code Server - maya'` and inspect the member's `logs\t3-serve.log`. Confirm the task principal, stored password and batch logon rights. Check that the member's `%APPDATA%\npm\t3.cmd` exists. If the task is running, check `Test-NetConnection 127.0.0.1 -Port 3774` and the member's `t3.cmd status`. Start or repair only the named member's task. |
| Port already in use | Check `Get-NetTCPConnection -LocalPort 8900 -State Listen`. Choose another assigned viewer port or use `--port 0` where supported and read the printed URL. For a T3 port conflict, have the administrator update the member's startup script and align the member's `service.env` and Connect setup. Leave the existing listener alone. |
| Waiting on the GPU lock | Report the job as queued. Keep the shared lock enabled and let the current job release it. If waiting persists after GPU activity ends, ask the administrator to check the common lock path and its permissions. Leave other jobs and the lock file alone. |
| Too little disk space or RAM | Check the output volume with `Get-PSDrive -Name E` and available RAM with `(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory` (KiB). The project pipeline refuses to run below 3 GB free disk. Budget space for labels, caches, runs, results and backups. Use a smaller pilot or the assigned SSD workspace. |
| Permission denied on a shared path | Verify `whoami` and have the administrator check the exact path and its ancestors with `icacls`. Shared Python and models need Read & Execute. Your workspace needs Modify. Keep model downloads and generated output out of shared directories. Coordinate any raw archive access with its owner. |
