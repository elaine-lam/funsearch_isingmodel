# funsearch_isingmodel
LLM project for REU2024 at UTK.<br />
FunSearch, an algorithm developed by Google in 2023, combines a pre-trained Large Language Model (LLM) with genetic programming principles to generate potential solutions to hard problems within mathematics. The LLM generates heuristics for solving the problem in the form of computer programs, allowing high-performing programs to be examined for insight into how the solutions work. We implement FunSearch using Meta's Llama3 to a problem within physics—finding the ground state of the Ising model. This commonly used model represents electron spin as a lattice of sites that each have two potential states. The energy of a spin configuration is found as a function of individual states and their interactions. Despite the model's simplicity, the problem of finding the lowest energy configuration is extremely complex and analytical solutions exist in only a few cases. There are a wide variety of applications for finding the ground state of the Ising model, as many other combinatorial optimization problems can be reduced to this problem. Due to time and computational constraints our FunSearch-generated programs not made significant improvements simple user-written optimization algorithms. However, FunSearch shows promise for our problem and represents a large step forward in terms of developing explainable AI.

## Development Environment
Python version == 3.10.12 <br />
OS: Linux Ubuntu 22.04.4 LTS <br />
CPU: 13th Gen Intel(R) Core(TM) i7-13700K<br />
GPU: NVIDIA GeForce RTX 4070


## Installation Guidiline
1. Install the dependency inside the requirements.txt.
2. Install <a href = "https://ollama.com/download">Ollama</a>:<br />
    Please visit the official Ollama webite to install: https://ollama.com/download.<br/> Here's the command to install in Linux system<br />
    ```bash 
    curl -fsSL https://ollama.com/install.sh | sh
    ```
3. Use Ollama to pull llama3: ollama pull llama3

## Pre-Execution
Start the ollama server by the following command
```bash
ollama serve
```
If the server cannot open successfully, you can visit https://github.com/ollama/ollama/issues/690 for some solution

## Execution Guideline
To start the loop, run funsearch.py or core.py.
- The funsearch.py is inside the implement folders. 
    1. To execute the funsearch.py, open the required file directory in terminal
    2. run command: `python funsearch.py`
- The core.py is loacted at the code folder.
    - To change the executing path for the subprocess, please change the file path inside the subprocess.run function
    1. To execute the core.py, open the code file directory in terminal
    2. run command: `python core.py`

## Folder Structure
The developed code is based on the <a href="https://github.com/google-deepmind/funsearch">official funsearch code</a>.<br />
For the documentation, please refer to the 2D_implement as reference.
- `code implementation` contains our code implementation 
    - `code_manipulation.py`: no changes is mada on this file
    - `config.py`: the configuration for the program setting
    - `error_log.py`: self-defined error logging function
    - `evaluate.py`: self-defined evaluate function for the priority function generated for 2D and 3D Ising model. For the Cap Set evaluate, it was copy from the opfficial code.
    - `evaluator.py`: implement our own Sandbox class to evaluate the generated program from the LLM. Also added some codes to record all scorable program into a txt file.
    - `funsearch.py`: added a function to make it executable.
    - `main.py`: You can also run this file to start executing the code
    - `programs_database.py`: the temp database for storing the program generated during execution. Added our functions to backup the program objects to pickle files, which help to roll back into certain execution state. 
    - `sampler.py`: implement our own LLM class. We use Llama 3 8B as our LLM.
    - `timeout.py`: Self-defined exception class for timeout.
- data/backup
    - program_db_priority.pickle
- `errorlog` contains the error logged during execution.
- generatedCode
- testdata
    - 2D
    - 3D
    - cap
        - `extractcode.py`: is in every sub-folder for the testdata have one. This program helps turning the generated code into an excel file.
- `code.txt`: the information to embedding when constructing the LLM
- `core.py`: use a subprocess function to keep the funsearch program running. Prevent from the unexpected termination.
- `data2D.txt`: the data for evaluate.py to use to score the generated program.
