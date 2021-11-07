import os
import json

VIDEO_PATH = '../output/video'
file_lst = os.listdir(VIDEO_PATH)
print(file_lst)

for name in file_lst:
    FILE_PATH = "output1.json"

    with open(FILE_PATH, "r") as json_file:
        json_data = json.load(json_file)
        save_file = open(f'../results{name}.json', 'w')
        json.dump(json_data, json_file)

