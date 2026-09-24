# Oimodeler App
Simplest way to get in interferometric data modelisation ! 


## Description
**Oimodeler App** is an interactive graphical interface built on top of the Python library [Oimodeler](https://github.com/oimodeler/oimodeler), designed to **model OIFITS** interferometric data **without requiring any code line**.

 The application provides an intuitive environment where users can load datasets, construct parametric models (e.g. uniform disks, Gaussians, rings), combine multiple components, and explore parameter spaces interactively.

## Compatibility
- **Supported: Python 3.11, 3.12, or 3.13.** This isn't an arbitrary choice: every scientific package pinned in `requirements.txt` (numpy, scipy, astropy, pyarrow...) ships a prebuilt wheel for exactly this range today. Outside it — Python 3.9/3.10, or 3.14 and newer — `pip install` will try to compile one or more of them from source, which needs a C/C++/Fortran toolchain most machines don't have, and will likely fail. See [Troubleshooting](#troubleshooting) if that happens to you.
- **Tested with:** Python 3.11 (the version this project is developed and validated against — see `runtime.txt`).
- **Operating systems:** Windows, macOS, Linux.

## Quick installation
For users already comfortable with Python and the command line. A virtual environment is used here to avoid dependency conflicts with other Python projects on your machine — see the detailed guide below for the Windows equivalent of `source env_oim/bin/activate`.

```bash
python3 --version                # must be 3.11, 3.12 or 3.13 — see Compatibility below
git --version
mkdir OimodelerApp
cd OimodelerApp
git clone https://github.com/vdhpfleury/oimodeler_App.git
cd oimodeler_App
python3 -m venv env_oim          # use python3.11 (or 3.12/3.13) explicitly if 'python3' resolves to another version
source env_oim/bin/activate      # Windows: env_oim\Scripts\activate
pip install -r requirements.txt
python doctor.py                 # optional: verify the install before launching
streamlit run app.py
```
Your default web browser should automatically open and display the application interface.



## Installation
Step-by-step instructions if you are new to Python, or if the quick installation above ran into an issue. Each step lists what to expect — if something doesn't match, jump to [Troubleshooting](#troubleshooting).

### 1. Check your Python version
Open a terminal and run:

macOS / Linux:
```bash
python3 --version
```
Windows:
```powershell
python --version
```
**Expected:** a line such as `Python 3.11.5`. Any of **3.11, 3.12 or 3.13** works — see [Compatibility](#compatibility) for why this range matters. If you're outside it (an older 3.9/3.10, or a newer 3.14+), install one of the supported versions alongside your current one rather than relying on whichever `python3`/`python` currently points to:

- **Windows:** download the Python 3.11 (or 3.12/3.13) installer from [python.org/downloads](https://www.python.org/downloads/) — check **"Add python.exe to PATH"** during installation. Multiple versions can coexist; the `py -3.11` launcher picks a specific one.
- **macOS:** `brew install python@3.11`
- **Linux (Debian/Ubuntu):** `sudo apt install python3.11 python3.11-venv` (add the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) first if your distribution's repositories don't have it)

Then use that specific interpreter to create the virtual environment in step 4, e.g. `python3.11 -m venv env_oim` instead of `python3 -m venv env_oim`.

- [x] Python 3.11, 3.12 or 3.13, recognized by your terminal


### 2. Check that Git is installed

Run the following command in your terminal:
```
git --version
```

**Expected:** something like `git version X.X.X`

If Git is not installed, follow the installation instructions provided in the official documentation: [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). On Windows, the "Git for Windows" installer adds Git to your PATH automatically.

- [x] git on your computer



### 3. Clone the git repository
You may first create a directory for the application:
```
mkdir OimodelerApp
cd OimodelerApp
```
Then clone the repository:
```
git clone https://github.com/vdhpfleury/oimodeler_App.git
```
Move into the repository folder:

```
cd oimodeler_App
```
Verify that all files are present:

```
ls
```
(on Windows, use `dir` instead of `ls`)

**Expected:** you should see, among others, `app.py`, `requirements.txt`, `README.md`, and the `pages/`, `core/`, `services/` folders.

- [x] the `oimodeler_App` folder contains `app.py`


### 4. Create a virtual environment
Strongly recommended: it isolates this project's Python packages from the rest of your system, so its dependencies (numpy, scipy, astropy, streamlit, oimodeler...) never clash with another Python project on your machine.

macOS / Linux:
```bash
python3 -m venv env_oim
source env_oim/bin/activate
```
Windows (PowerShell):
```powershell
py -m venv env_oim
.\env_oim\Scripts\Activate.ps1
```
Windows (cmd.exe):
```cmd
py -m venv env_oim
env_oim\Scripts\activate.bat
```

If your default `python3`/`python`/`py` is outside 3.11–3.13 (see step 1), point the same commands at the specific version instead, e.g. `python3.11 -m venv env_oim` (macOS/Linux) or `py -3.11 -m venv env_oim` (Windows).

**Expected:** your terminal prompt now shows an `(env_oim)` prefix. You will need to re-run the activation command every time you open a new terminal, before steps 5 and 7.

- [x] the virtual environment is created and active


### 5. Install the Python dependencies
With the virtual environment active, install everything the app needs:
```
pip install -r requirements.txt
```
This also installs `oimodeler` itself, directly from its GitHub repository — no separate install needed. If `pip` is not recognized, refer to the official documentation: [doc here](https://pip.pypa.io/en/stable/installation/).

**Expected:** the last line reads something like `Successfully installed ...`, with no persistent red `ERROR`. This step downloads about twenty scientific packages and can take a few minutes.

- [x] `pip install` finished without errors


### 6. Verify the installation (optional but recommended)
Before launching the app, you can run a quick diagnostic that checks your Python version, that every dependency imports correctly, and that the project files are in place:
```
python doctor.py
```
**Expected:** a report ending with `All checks passed. You can now run: streamlit run app.py`. If it reports a `[FAIL]`, see [Troubleshooting](#troubleshooting).


### 7. Run the app
To start the application, run the following command from the repository directory (with the virtual environment active):
```
streamlit run app.py
```

**Expected:** the terminal prints a local URL (`http://localhost:8501`) and your default web browser automatically opens on the application, with its five tabs: Overview, Component Explorer, Data, Modelling, Fitting. To stop the app, go back to the terminal and press `Ctrl+C`.

**Then you're done!**


## Troubleshooting

**`python3: command not found` / `'python' is not recognized`**
Python isn't installed, or wasn't added to your PATH. Reinstall it from [python.org/downloads](https://www.python.org/downloads/) — on Windows, check "Add python.exe to PATH" — then close and reopen your terminal.

**`git: command not found`**
Git isn't installed. Follow [git-scm.com's installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git); on Windows, use the "Git for Windows" installer, which adds Git to your PATH automatically.

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
