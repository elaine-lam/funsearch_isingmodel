#use to process the reponse by gpt
import re 
with open("tempStore.txt", "r") as file1:
    temp = file1.read()

substr = "```"
sub_location = temp.find(substr)
temp = temp[int(sub_location)+3:]
sub_location = temp.find(substr)
temp = temp[:int(sub_location)]

def_location = temp.find("def ")
temp = temp[int(def_location):]
print(temp)