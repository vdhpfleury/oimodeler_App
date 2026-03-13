

5. Run the application

To start the application, run the following command from the repository directory:

streamlit run 3_test.py

Your default web browser should automatically open and display the application interface.

(Add screenshot here)

User Guide

In progress.

## Contact

Oimodeler App






# Oimodeler App
Simplest way to get in interferometric data modelisation ! 


## Description
**Oimodeler App** is an interactive graphical interface built on top of the Python library [Oimodeler](https://github.com/oimodeler/oimodeler), designed to **model OIFITS** interferometric data **without requiring any code line**.

 The application provides an intuitive environment where users can load datasets, construct parametric models (e.g. uniform disks, Gaussians, rings), combine multiple components, and explore parameter spaces interactively.

## Installation
### 1. Check your Python version
Open a terminal and run:

```
python3 --verison
```
If your Python version is 3.9 or higher, you can proceed. Otherwise, install a newer version from the official Python website: (see [python website](https://www.python.org/downloads/)). After installation, make sure Python is added to your system PATH environment variable.

- [x] Python > 3.9


### 2. Check that Git is installed

Run the following command in your terminal:
```
git --verison
```

This should return something like: git version X.X.X

If Git is not installed, follow the installation instructions provided in the official documentation: [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

- [x] git on your computer



### 3. Clone the git repository.
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
You should see something similar to:
- 3_test.py   
- extlaws 
- README.md  
- requirements.txt

- [x] environement ready

### 4. Set up the Python dependencies
The application requires several Python packages to run.
You can install them using:
```
pip install -r requirements.txt
```
If pip is not installed on your system, refer to the official documentation:
(doc [here](https://pip.pypa.io/en/stable/installation/)).
This install all required dependencies listed in the requirements.txt file.

Note: using a virtual environment is recommended but optional.

**Then your done !**

### 5. Run the app
To start the application, run the following command from the repository directory:
```
streamlit run 3_test.py
```

Your default web browser should automatically open and display the application interface.


## User Guide
[in progres...]

## Contact 
oimodeler app      : valentin.fleury@oca.eu
