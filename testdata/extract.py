# with open("./testdata/generateHvScorePrifun.txt", 'a') as file: 
#     file.writelines('#score: ' + str(scores_per_test) + '\n')
#     file.writelines('#island_id: ' + str(island_id) + '\n')
#     file.writelines('#version_generated: ' + str(version_generated) + '\n')
#     file.writelines('program:\n' + str(program) + '\n\n\n')

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

df1 = pd.DataFrame(temp.items(), index=range(len(temp)), columns=['program', 'score'])
df1.to_excel("output.xlsx")  