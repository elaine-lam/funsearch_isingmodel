# with open("./testdata/generateHvScorePrifun.txt", 'a') as file: 
#     file.writelines('#score: ' + str(scores_per_test) + '\n')
#     file.writelines('#island_id: ' + str(island_id) + '\n')
#     file.writelines('#version_generated: ' + str(version_generated) + '\n')
#     file.writelines('program:\n' + str(program) + '\n\n\n')

df = {}
with open("./testdata/generateHvScorePrifun.txt", 'r') as file:
    dataset = file.read()

scores = []
for data in dataset.split("#score: ")[1:]:
    score = data[data.find("'data2D.txt': ")+14:]
    score = score[:score.find("\n")-1]
    scores.append(int(scores))
    
scores.sort()
print(scores)