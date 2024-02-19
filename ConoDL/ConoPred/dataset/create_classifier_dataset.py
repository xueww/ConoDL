import json
import random

def read_file(path, num):
    with open(path, 'r')as f:
        data = f.read().split('\n')
    return [[i, num]for i in data]

def write_file(path, train_data, test_data):
    with open(path, 'w')as f:
        f.write(json.dumps({'train': train_data, 'test': test_data}))


zheng = read_file('positive.txt', 1)
fu = read_file('negative.txt', 0)
all_dataset = positive+negative
random.shuffle(all_dataset)
train = all_dataset[:int((len(all_dataset)*0.8))]
test = all_dataset[int((len(all_dataset)*0.8)):]

write_file('train_classifier.json', train, test)