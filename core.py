import subprocess

result =  -1

#start an infinte loop to execute program
while True:
    result = subprocess.run(["python", "./funsearch_official_code/funsearch.py"], capture_output=True, text=True)
    print(result)

