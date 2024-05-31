⭐ If you find this repository helpful, please consider giving it a ⭐ here on GitHub (click the star button in the top right corner) 
It's a quick way to show support for this openly available code. ⭐

![](resources/manning_publications_logo.png)

# Transformers in Action
This is the corresponding code for the book Transformers in Action

The book covers:

* Part 1 Introduction to transformers 
  * The need for transformers 
  * A deeper look into transformers 
* Part 2 Transformers for fundamental NLP tasks 
   * Text summarization
   * Machine translation
   * Text classification
* Part 3 Advanced models and methods 
   * Text generation  
   * Prompting and zero to few shot learning  
   * Multimodal models 
   * Optimize and evaluate large language models
   * Ethical and responsible Tlarge language models
 

## Instructions and Navigation
All of the code is organized into folders. Each folder starts with `CH` followed by the chapter number. For example, CH03.
The notebooks are then organized as follows: `ch03_text_summarization_eval.ipynb`, where `ch03` indicates the chapter
and `text_summarization_eval` what is done in the notebook. 

## Virtual Envrionment

The provided bash script `create_env.sh` automates the process of creating a Python virtual environment using either conda or pipenv, 
installing the required packages from a `requirements.txt file`. To use the script run `bash create_env.sh` in your 
terminal on Microsoft Windows (with WSL), Apple macOS, or Linux operating systems.

<span style="color:red">
NOTE: A virtual environment is not necessary for the notebooks in this repository, as they are designed to be 
run on a cloud service with GPU support. Therefore, the provided instructions for creating a virtual environment are 
more for reference and general guidance than a strict requirement. </span>
<br />
<br />

However, if your machine is powerful enough and if you really want to create a local environment, please run the following commands from the console:

~~~
# Clones the repo in the current folder
git clone https://github.com/Nicolepcx/Transformers-in-Action.git

# Enters to the folder of the cloned project
cd Transformers-in-Action

# Creates a virtual environment
python -m venv venv

# Activates the virtual environment
source venv/bin/activate

# Installs the dependencies
pip install -r requirements_LOCAL.txt --use-pep517

# Opens a new tab in the browser to run the notebooks 
jupyter notebook
~~~

Then, feel free to use every notebook with the prefix 'LOCAL' appended to its name.

## Running the Notebooks

Every notebook contains buttons so that the notebook can be oppend and run on the chosen cloud service like this:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()   [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)]() 
__NOTE:__ As of the currently used version of Hugging Face datasets there is an issue
on Kaggle, which is why it is, for now, not supported for the notebooks. 


Each notebook is connected with this Github repo, meaning by running a notebook, it will automatically clone the repo, so you can easily access all resources outside the notebook.
Like customs functions and classes as well as utility functions to automatically install the requirements per chapter: 


```
!git clone https://github.com/Nicolepcx/Transformers-in-Action.git

current_path = %pwd
if '/Transformers-in-Action' in current_path:
    new_path = current_path + '/utils'
else:
    new_path = current_path + '/Transformers-in-Action/utils'
%cd $new_path
```
__NOTE:__ You need to run the notebooks with a GPU. 

## Project structure

```
├── LICENSE
├── README.md             <- The top-level README for developers using this project.
├── CH02                  <- Per chapter folder with Jupyter notebooks.
    ├── [name].ipynb      <- Jupyter notebooks with naming as mentioned above.
├── CH03                  <- Per chapter folder with Jupyter notebooks.
...                       <- Same structure for all chapters.
├── utils                 <- Custom classes and functions and utility functions.
├── resources             <- Some miscellaneous resources such as the logo.

```
