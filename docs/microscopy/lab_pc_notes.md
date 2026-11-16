# Running the microscopy pipeline on the lab PC: lessons

Lessons from the first day on the lab PC (Windows 10, `DESKTOP-I8INR24`, 2026-11-15), while analysing the KLY / EPEC / TBL PI acquisition of 12–13 Nov 2026. Each item cost real time once. They are written so the next agent, or person, does not pay for them again.

## Machine and layout

- **Disks.** C: is a 500 GB NVMe SSD with little free space. E: is a 1 TB **spinning hard disk** (Seagate ST1000DM003) that holds data, environments and outputs under `E:\microscopy\`. D: is a shared USB SSD that new acquisitions arrive on.
- **Two data-heavy jobs on E: at once thrash the disk.** Indexing 87 GB dropped to 11 MB/s while a benchmark read the same disk. Run one I/O-heavy job at a time, or move the active acquisition to an SSD.
- **Layout under `E:\microscopy\`:** `raw\` (acquisitions, never written to), `work\<experiment>\` (projects, runs, results, notebooks), `code\` (frozen code exports, see below), `env\` (main Python 3.12 environment), `env-gpu\` (Omnipose and CUDA PyTorch), `logs\`.

## Python environments

- **Anaconda on this PC is unusable.** `C:\ProgramData\Anaconda3` is conda 22.9 on Python 3.9. Its base is read-only, conda crashes, and its pip fails on a permission error. Use `uv` (installed at `%USERPROFILE%\.local\bin\uv.exe`) with `UV_PYTHON_INSTALL_DIR` and `UV_CACHE_DIR` on E:.
- **Main environment:** `uv venv --python 3.12 E:\microscopy\env`, then `uv pip install -e <repo>[test] scikit-image jupyterlab`.
- **GPU environment** (`E:\microscopy\env-gpu`, Python 3.10):
  - Installing `omnipose` replaces CUDA PyTorch with a CPU build. Reinstall `torch` from `https://download.pytorch.org/whl/cu126` afterwards, and check that `torch.cuda.is_available()` returns True.
  - Omnipose 1.1.4 imports `aicsimageio`, which needs `tifffile==2023.2.28`. Newer versions fail with `'_TIFF' object has no attribute 'RESUNIT'`.
  - It also needs `numpy<2`.
- **Read text files as UTF-8 explicitly.** Windows' default encoding (cp1252) fails on files containing characters such as ρ.

## Running long jobs: detach them

- **A Claude Code session restart stops every background task it started.** On 15 Nov this killed the review viewer while the user was tagging from their Mac, and killed an hour-long replay run.
- **Start servers and multi-hour runs detached.** Use a small `.ps1` launcher started with `Start-Process ... -WindowStyle Hidden`, writing its log and a status file next to the data. Examples: `work\KLYEPEC\review\start_viewer.ps1` and `work\KLYEPEC\rescue_job.ps1`. Agents may still watch them; they should not own them.
- **Freeze the code a production run uses.** The main environment has the repo checkout installed in editable mode, and agents edit that checkout on their own branches. A run that imports from it mid-edit would mix code versions. Export the exact commit and run from that:
  ```
  git archive <commit> microscopy | tar -x -C E:\microscopy\code\frozen-<sha>
  set PYTHONPATH=E:\microscopy\code\frozen-<sha>
  ```
  Spawned worker processes inherit `PYTHONPATH`.
- **Resuming from saved labels still re-reads every raw image.** It recomputes quality numbers and the cell list for each frame: about 30 minutes for 5,304 frames from the hard disk. Reusing those per-frame results as well is an open speed-up.

## Windows-specific traps

- **`.js` served as `text/plain`.** Python's `mimetypes` reads the Windows registry, which on this PC maps `.js` to `text/plain`. Browsers then refuse the viewer's scripts and the page sits at "Connecting...". The viewer now sets content types explicitly. **Check web changes in a real browser**: tests that fetch files with `urllib` did not catch this.
- **`-ErrorAction SilentlyContinue` hides "Access denied".** A non-administrator cannot read firewall rules, so a check for the viewer's rule reported "missing" when it existed. Check the error rather than suppressing it, or ask the user to check from an administrator shell.
- **PowerShell variable names are case-insensitive.** `$S` and `$s` are the same variable.
- **Symlinks need Developer Mode or administrator rights.** One test skips without them.
- **`os.nice` and `os.sysconf` do not exist on Windows.** Use `microscopy/system.py`. When calling Win32 functions through ctypes, declare `restype` and `argtypes`. Without them, the 64-bit process handle is truncated and the call fails silently.
- **CPU priority.** Workers run at below-normal priority so the PC stays usable.

## Reviewing from another computer

- **The viewer runs on the PC in network mode** (`--host 0.0.0.0 --allow-client ...`) with an access key (see the README).
- **The PC's firewall needs one inbound rule**, created by an administrator, for ports 8890–8899 from the campus subnet and Tailscale's range (`100.64.0.0/10`).
- **The access key persists** in `%USERPROFILE%\.lab-microscopy\viewer_access_key`. Delete that file to revoke old links.
- **Display images:** about 500 KB per frame as PNG, about 120 KB as JPEG (`--image-format jpeg`). Use JPEG off campus.
- **The viewer's default phase contrast is fixed from frame 0.** Dense late frames therefore look washed out. Use "Auto phase contrast" before concluding a frame is bad. A brightness surge on 12–13 Nov (Pos16, frames 129–133) was real but harmless for segmentation.

## Agents

- **Codex** is available as a delegated agent through T3 Code once the Codex CLI is installed and logged in. It can hit the ChatGPT plan's usage limit mid-task. Tell it to commit work-in-progress (marked WIP) before stopping.
- **Several agents share one checkout.** Give each its own branch, and do not switch the checkout's branch while another agent works in it. T3's worktree isolation needs the T3 project to be the repo itself.
