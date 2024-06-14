# execute_code.py
import numpy as np
# Define a function to execute code from a file
def execute_code_from_file(file_path):
    with open(file_path, "r") as file:
        code = file.read()
        codeObject = compile(code, 'sumstring', 'exec')
        exec(codeObject, globals())

# Example usage
if __name__ == "__main__":
    file_path = "convertdata.txt"  # Change this to the path of your text file
    execute_code_from_file(file_path)
