import os
submit_path = "/home/data/zhuchuanbo/Documents/competition/JHT/src/run/HYSZ-华中科技大学-陈进才"
result_path = "/home/data/zhuchuanbo/Documents/competition/JHT/src/run/standardID.txt"

files = os.listdir(submit_path)

submits = {}

for file in files:
    submits[file.split("_")[1]] = file.split("_")[2].replace(".txt","")

answers = {}

with open(result_path, "rb") as f:
    lines = f.readlines()
    for line in lines:
        line = str(line, "utf-8")
        line = line.replace("\r\n", "")
        line = line.split(" ")
        answers[line[0]] = line[1]
        
# 正确率：已知样本中，正确的个数
# 拒识律：未知样本中，正确的个数

correct_number, none_number = 0, 0
for key, value in answers.items():
    if value != '9999':
        if submits[key] == value:
            correct_number += 1
    else:
        if submits[key] == '9999':
            none_number += 1

print("{}-{}".format(correct_number, none_number))
print((correct_number / 121) * 0.6 + (none_number / 21) * 0.3)
    