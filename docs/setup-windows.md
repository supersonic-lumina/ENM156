# Setup on Windows (WSL)
 
> **On Linux?** Skip this page and follow the [Linux setup](setup-linux.md).
>
> Back to the [README](../README.md).
 
This project is developed on Linux. On Windows, you run it inside **WSL 2** (Windows Subsystem
for Linux), which gives you a real Linux shell on your own machine. This page covers the
Windows-specific parts. When you're done here, you continue with the Linux guide, because
everything after this runs inside WSL exactly as on Linux.
 
## Contents
 
1. [Install WSL](#1-install-wsl)
2. [Open a Linux shell](#2-open-a-linux-shell)
3. [Keep the project in the Linux filesystem](#3-keep-the-project-in-the-linux-filesystem)
4. [Configure line endings](#4-configure-line-endings)
5. [GPU support (optional)](#5-gpu-support-optional)
6. [Performance (optional)](#6-performance-optional)
7. [Editor (optional)](#7-editor-optional)
Then: [Continue with the Linux guide](#continue-with-the-linux-guide)
 
## 1. Install WSL
 
Open **PowerShell as administrator** and run:
 
```powershell
wsl --install -d Ubuntu-24.04
```
 
Restart the computer if it asks you to. When Ubuntu starts for the first time, it asks you to
create a username and a password.
 
- The username is yours to choose. It's only used inside WSL.
- You'll type the password every time you use `sudo`. Choose one you can remember and that you
  don't use anywhere important.
If `wsl --install` only prints a help text, WSL is outdated: run `wsl --update` and try again.
If it reports that virtualization is disabled, enable virtualization (Intel VT-x or AMD-V /
SVM) in your BIOS or UEFI settings.
 
## 2. Open a Linux shell
 
Either:
 
- open **Ubuntu** from the Start menu, or
- run `wsl.exe` in PowerShell or Windows Terminal.
You should see a prompt like `user@computer:~$`. Everything from here on is typed into this
shell, not into PowerShell.
 
## 3. Keep the project in the Linux filesystem
 
Clone the repository under your Linux home folder, for example `~/projects`:
 
```bash
mkdir -p ~/projects
cd ~/projects
```
 
Don't put it under `/mnt/c/...` (your Windows drives). Reading and writing files there from
WSL is much slower, which hurts when training loads thousands of audio clips. You can still
open your Linux files from Windows Explorer by typing `\\wsl$\Ubuntu-24.04` in the address bar.
 
## 4. Configure line endings
 
Windows and Linux end lines differently, which causes noisy diffs. Tell git to keep Linux line
endings:
 
```bash
git config --global core.autocrlf input
```
 
## 5. GPU support (optional)
 
To train on an NVIDIA GPU:
 
1. Install the current NVIDIA driver **on Windows**. Don't install a Linux driver inside WSL;
   WSL uses the Windows one.
2. In the WSL shell, check that it's visible:
```bash
nvidia-smi
```
   You should see a table with your GPU. If the command isn't found, update your Windows
   driver and run `wsl --update` and `wsl --shutdown` from PowerShell, then reopen Ubuntu.
 
Without a supported GPU, everything still runs on the CPU, only slower.
 
## 6. Performance (optional)
 
By default, WSL may use only about half of your computer's RAM. Training holds many audio clips
in memory, so you may need more. Create or edit `C:\Users\<your-name>\.wslconfig` in Windows:
 
```ini
[wsl2]
memory=16GB
```
 
Pick a value that leaves enough for Windows, then run `wsl --shutdown` in PowerShell and
reopen Ubuntu. You can also lower `train.max_train` in your `config.local.toml` instead (see
the Linux guide).

Some useful settings you might want to set in .wslconfig, especially if you run on the cpu:
| Setting | Description |
|-:|:-|
| memory | Maximum RAM available to WSL |
| processors | Number of CPU cores available |
| swap | Size of swap space |
| swapFile | Location of swap file |

Example config:
```ini
[wsl2]
memory=8GB
processors=6
swap=4GB
```
 
## 7. Editor (optional)
 
If you use VS Code, install the **WSL** extension. From the project folder in the WSL shell,
run:

```bash
code .
```
 
VS Code opens on Windows but works on the files and terminal inside WSL.

This generally improves performance and is the recommended workflow.
 
## Continue with the Linux guide
 
You now have a Linux shell, a place for the code, and (optionally) GPU access. Continue with
**[Linux setup](setup-linux.md)**.
Everything there is typed into the WSL shell.