# execute_code.py
import numpy as np
# Define a function to execute code from a file
def execute_code_from_file(file_path):
    with open(file_path, "r") as file:
        code = file.read()
        codeObject = compile(code, 'sumstring', 'exec')
        exec(codeObject, globals())
        N = 5
        J = 1.0
        ground_state, min_energy = ising_ground_state(N, J)
        print("Ground state:", ground_state)
        print("Minimum energy:", min_energy)

# Example usage
if __name__ == "__main__":
    file_path = "convertdata.txt"  # Change this to the path of your text file
    execute_code_from_file(file_path)
