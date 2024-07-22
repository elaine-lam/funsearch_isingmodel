import numpy as np
from evaluate import evaluate
import pandas as pd
from datetime import date
import pickle




with open('data3D.txt', 'rb') as handle:  # import data
    dataset3D = pickle.loads(handle.read())

with open("./testdata/3D/generatedPrifun3D.txt", 'r') as file:
  dataset = file.read()

temp_programs = []
temp_scores = []
temp_stdev = []
for data in dataset.split("#score: ")[1:]:
  program = data[data.find("def priority"):]
  while True:
    if program[-1] == '\n':
      program = program[:-1]
    else:
      break
  score = data[:data.find("\n")].strip()
  if data.find("standard deviation: ") != -1:
    stdev = data[data.find("standard deviation: ")+20:]
    stdev = stdev[:stdev.find("\n")-1]
    # exec(program)
    # score2, stdev = evaluate(dataset3D, priority) #type: ignore
    # assert(float(score) == score2)
  else:
    stdev = 0
  temp_programs.append(program)
  temp_scores.append(float(score))
  temp_stdev.append(float(stdev))



file2 = "./testdata/3D/2024-07-22generateHvScorePrifun3D.txt"
with open(file2, 'r') as file:
  dataset2 = file.read()
for data in dataset2.split("#score: ")[1:]:
  program = data[data.find("def priority"):]
  while True:
    if program[-1] == '\n':
      program = program[:-1]
    else:
      break
  score = data[data.find("'data3D.txt': ")+14:]
  score = score[:score.find("\n")-1]
  if data.find("standard deviation: ") != -1:
    stdev = data[data.find("standard deviation: ")+20:]
    stdev = stdev[:stdev.find("\n")-1]
    # exec(program)
    # score2, stdev = evaluate(dataset3D, priority) #type: ignore
    # assert(float(score) == score2)
  else:
    stdev = 0
  temp_programs.append(program)
  temp_scores.append(float(score))
  temp_stdev.append(float(stdev))

df = pd.DataFrame({'program': temp_programs, 'score': temp_scores, 'standard deviation': temp_stdev})
df = df.sort_values(by = 'score')
df = df.reset_index(drop = True)
print(df)
name = "./generatedCode/3D/" + date.today().strftime("%m-%d") + "code.xlsx"
df.to_excel(name)



with open("./testdata/3D/generatedPrifun3D.txt", 'w') as file:
  for i in range(len(df)):
    file.writelines("#score: " + str(df["score"].loc[i]) + '\n')
    file.writelines("#standard deviation: "+str(df["standard deviation"].loc[i]) + '\n')
    file.writelines("#program:\n" + str(df["program"].loc[i]) + '\n\n\n')