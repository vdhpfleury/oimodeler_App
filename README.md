# Oimodeler App
Simplest way to get in interferometric data modelisation ! 


## Description
**Oimodeler App** is an interactive graphical interface built on top of the Python library [Oimodeler](https://github.com/oimodeler/oimodeler), designed to **model OIFITS** interferometric data **without requiring any code line**.

 The application provides an intuitive environment where users can load datasets, construct parametric models (e.g. uniform disks, Gaussians, rings), combine multiple components, and explore parameter spaces interactively.

## Installation
### 1. Ensure you have the good python version. 
=> write in your terminal : 

```
python3 --verison
```

If the version is larger than 3.9 your fine otherwise download a python version > 3.9. (see [python website](https://www.python.org/downloads/)). After dowload don't forget to add python to your PATH env. variable.

- [x] Python > 3.9

### 2. Ensure you have the git command on your computer. 
=> write in your terminal : 

```
git --verison
```

should return : git version XXX

Otherwise install it by following Git Documentation [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) according to your operating system (Linux/OS)

- [x] git on your computer



### 3. Clone the git repository.

Eventually you can create a directory Oimodeler_App move to this folder.

```
mkdir OimodelerApp
cd OimodelerApp
```

Then clone the repository with the command :


```
git clone https://github.com/vdhpfleury/oimodeler_App.git
```

After the clonning succed go to the folder oimodeler_App

```
cd oimodeler_App
```

And tchek all file are present with the ls command : 

```
ls
```
should return the following list of file/folder : 
- 3_test.py   
- extlaws 
- README.md  
- requirements.txt

-[x] envirnment ready

### 4. Set up the Python dependencies
In order to work properly, this app need packages & libraries. To install it you can either create a specific virtual environement (see doc here to do so) but can also install all the dependencies directly on your computer thanks to the command : 
```
pip install -r requirements.txt
```
Of course you need the pip command on your computer. (doc [here](https://pip.pypa.io/en/stable/installation/))

Then your done !

### 5. Run the app
Each time you want to run the app, you just have to run this command line in your terminal in the repository were the file .py is store : 
```
streamlit run 3_test.py
```

A window should be open in your favorite brawser with the welcome app page.
add image

(Comment ajouter sa à l'environnement de l'ordinateur pour qu'en écrivant simplement oimodeler_app dans le terminal cela lance l'application ?)





## User Guide
[in progres...]

## Contact 
oimodeler app      : valentin.fleury@oca.eu
