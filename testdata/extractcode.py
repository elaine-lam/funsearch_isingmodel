from datetime import date
import pandas as pd

with open("./testdata/generatedPrifun.txt", 'r') as file:
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

file2 = "./testdata/2024-07-23generateHvScorePrifun.txt"
with open(file2, 'r') as file:
  dataset2 = file.read()

for data in dataset2.split("#score: ")[1:]:
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
name = "./generatedCode/" + date.today().strftime("%m-%d") + "code.xlsx"
df1.to_excel(name)  

with open("./testdata/generatedPrifun.txt", 'w') as file:
  for key, value in temp.items():
    file.writelines("#score: {'data2D.txt': " + str(value) + '}\n')
    file.writelines("program:\n" + key + '\n\n\n')
