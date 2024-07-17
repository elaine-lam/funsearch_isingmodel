import pandas as pd

df = pd.read_excel('./generatedCode/07-11code.xlsx')[:111]
name = "code.txt"
name2 = "script.txt"

with open(name, 'w') as file:
    file.writelines('{')
    for i in range(len(df["program"])):
        if i == len(df["program"])-1:
            with open(name2, 'w') as file1:
                file1.writelines(df["program"][i] + '\n')
            break
        file.writelines(',{"prompt":"#score: '+ str(df["score"][i]))
        file.writelines('","completion":"' + df["program"][i] + '"}\n')
    file.writelines('}')
    

