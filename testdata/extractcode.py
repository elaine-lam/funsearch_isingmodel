

import pandas as pd

with open("./testdata/generateHvScorePrifun.txt", 'r') as file:
    dataset = file.read()

temp = {}
for data in dataset.split("#score: ")[1:]:
    program = data[data.find("def priority"):]
    while True:
        if program[-1] == '\n':
            program = program[:-1]
        else:
            break
    score = data[data.find("'data2D.txt': ")+14:]
    score = score[:score.find("\n")-1]
    temp.__setitem__(program, float(score))

sorted_temp = sorted(temp.items(), key=lambda x:x[1])

df1 = pd.DataFrame(sorted_temp, index=range(len(sorted_temp)), columns=['program', 'score'])
df1.to_excel("7-3code.xlsx")  