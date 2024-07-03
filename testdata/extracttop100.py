import pandas as pd

df = pd.read_excel('7-3code.xlsx')[:101]
name = "code.txt"
name2 = "script.txt"

with open(name, 'w') as file:
    for i in range(len(df["program"])):
        if i == len(df["program"])-1:
            with open(name2, 'w') as file1:
                file1.writelines(df["program"][i] + '\n')
            break
        file.writelines("#score: " + str(df["score"][i]))
        file.writelines("\n" + df["program"][i] + '\n\n')
    

