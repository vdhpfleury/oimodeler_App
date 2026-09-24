# Oimodeler App
Simplest way to get in interferometric data modelisation ! 


## Description
**Oimodeler App** is an interactive graphical interface built on top of the Python library [Oimodeler](https://github.com/oimodeler/oimodeler), designed to **model OIFITS** interferometric data **without requiring any code line**.

 The application provides an intuitive environment where users can load datasets, construct parametric models (e.g. uniform disks, Gaussians, rings), combine multiple components, and explore parameter spaces interactively.

 ## Quick Access 
- Online version : [Oimodeler App](https://oimodeler-app.streamlit.app/)
- Local version : **One Line Command** to install Oimodeler-App on your computer
  
Linux/macOS:
```bash
curl -fsSL https://raw.githubusercontent.com/vdhpfleury/oimodeler_App/main/installer/install.sh | bash
```
Windows (PowerShell):
```powershell
irm https://raw.githubusercontent.com/vdhpfleury/oimodeler_App/main/installer/install.ps1 | iex
```

**Desktop shortcut.** => relaunch the app later without opening a terminal !
- **Linux:** an "OIModeler App" entry is added to your applications menu, and copied to `~/Desktop` if that folder exists. Some file managers (e.g. GNOME Files) require a one-time right-click → "Allow Launching" on a new desktop shortcut before double-click works — that's an OS security step, not a bug.
- **macOS:** an `OIModeler App.command` file is placed on your Desktop; double-click it to relaunch.
- **Windows:** an "OIModeler App" shortcut is added to your Desktop.


### Note on the installation process
This is one line command is the recommended way to get OIModeler App running locally: it finds a suitable Python, downloads the app, creates an isolated environment, installs every dependency, verifies the install, and launches it. You still need Python 3.11, 3.12 or 3.13 and Git available on your machine first (see [Compatibility](#compatibility)) — the installer checks for both and tells you exactly what's missing if something is.

If PowerShell blocks that command (execution policy), download `installer/install.ps1` and run `powershell -ExecutionPolicy Bypass -File installer\install.ps1` instead.

Run from an empty folder and it downloads the app for you; run it from inside an existing checkout (`installer/install.sh` or `installer\install.ps1`) and it reuses that checkout instead. Re-running it later is safe — it reuses the environment it already created.

Your default web browser should then automatically open and display the application interface. If a step fails, the installer names the problem — for the fixes, see [Troubleshooting](#troubleshooting).




## Compatibility
- **Supported: Python 3.11, 3.12, or 3.13.** See [Troubleshooting](#troubleshooting) if you have error.
- **Tested with:** Python 3.11
- **Operating systems:** Windows, macOS, Linux

  
## Troubleshooting

*The [one-command installer](#install-and-run--one-command) checks for Python and Git upfront and names exactly what's missing before doing anything else — if you ran it, its own message already told you which of the two entries below applies. The entries below are for the manual/advanced path, or for reading the installer's message in more detail.*

**`python3: command not found` / `'python' is not recognized`**
Python isn't installed, or wasn't added to your PATH. Reinstall it from [python.org/downloads](https://www.python.org/downloads/) — on Windows, check "Add python.exe to PATH" — then close and reopen your terminal.

**`git: command not found`**
Git isn't installed. Follow [git-scm.com's installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git); on Windows, use the "Git for Windows" installer, which adds Git to your PATH automatically. Git is needed even with the one-command installer: `pip` uses it internally to fetch the `oimodeler` library from its GitHub repository.

**`ModuleNotFoundError: No module named 'oimodeler'` (or any other package)**
Your virtual environment likely isn't active, or step 5 didn't complete. Activate it (step 4), then re-run `pip install -r requirements.txt` and `python doctor.py` to confirm.

**`streamlit: command not found`**
Same cause as above: activate the virtual environment (step 4) before running `streamlit run app.py`.

**`pip install -r requirements.txt` fails while *building* `pyarrow`, `astropy`, `numpy` or `scipy` from source** (errors mentioning `cmake`, `Arrow`, a missing `FindArrow.cmake`, or a C/C++/Fortran compiler)
This means the virtual environment was created with a Python version outside the supported 3.11–3.13 range (see [Compatibility](#compatibility)) — most often a brand-new release (3.14+) or an old one (3.9/3.10) for which these packages don't publish a prebuilt wheel. pip then falls back to compiling them, which needs system libraries this project doesn't ask you to install, and the build fails. Fix: install Python 3.11, 3.12 or 3.13 (see the callout in step 1), delete the broken environment, and recreate it with that interpreter:
```
rm -rf env_oim                   # Windows: rmdir /s /q env_oim
python3.11 -m venv env_oim       # Windows: py -3.11 -m venv env_oim
source env_oim/bin/activate      # Windows: env_oim\Scripts\activate
pip install -r requirements.txt
```

**`pip install -r requirements.txt` fails for another reason**
Check your Python and pip versions:
```
python --version
pip --version
```
Make sure you're on a supported Python version (3.11–3.13, step 1) and connected to the internet — `requirements.txt` installs `oimodeler` directly from GitHub via Git, so step 2 (Git installed) also matters here. Never run `pip install` with `sudo`; use a virtual environment instead, or add `--user` to the command if you skipped one.

**Port 8501 is already in use**
Another process (or a previous Streamlit run) is already using it. Start the app on a different port:
```
streamlit run app.py --server.port 8502
```
then open `http://localhost:8502`.

**The app starts but crashes immediately, or `python doctor.py` reports a `[FAIL]`**
Run `python doctor.py` from the project's root folder for a full report of what's missing. If a specific dependency fails to import, re-run `pip install -r requirements.txt` inside the active virtual environment. If the problem persists, copy the error from the terminal and contact us (see Contact below).


## User Guide
[in progres...]

## Contact 
oimodeler app      : valentin.fleury@oca.eu
