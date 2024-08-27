import json
import matplotlib.pyplot as plt


with open("data/split_keys.json") as json_file:
    keys = json.load(json_file)

train_keys = keys["train"]
val_keys = keys["val"]
print(len(train_keys), len(val_keys))

line_count = []

above_60 = 0
for train_key in train_keys:
    with open("data/training/"+train_key+".json") as json_file:
        item = json.load(json_file)
        line_count.append(item['program'].count('\n'))
        if item['program'].count('\n')>59:
            above_60 += 1


for val_key in val_keys:
    with open("data/training/"+train_key+".json") as json_file:
        item = json.load(json_file)
        line_count.append(item['program'].count('\n'))
        if item['program'].count('\n')>59:
            above_60 += 1


plt.hist(line_count)

plt.savefig('line_hist.png')

print(above_60)

left = ['36fdfd69', '6aa20dc0', 'd22278a0', 'b7249182', '9d9215db', '264363fd', '234bbc79', '2dd70a9a', 'a64e4611', '7837ac64', '97a05b5b']

for left_program in left:
    with open("data/training/"+left_program+".json") as json_file:
        item = json.load(json_file)
        print(item['program'])
        n_line = item['program'].count('\n')
        print(f'---------------{n_line}------------------')