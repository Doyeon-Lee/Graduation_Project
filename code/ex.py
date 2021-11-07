import os
import re
import json

VIDEO_PATH = '../media/non-violence'
file_lst = os.listdir(VIDEO_PATH)
file_lst = [re.sub('.mp4', '', i) for i in file_lst]

for name in file_lst:
    FILE_PATH = f'../output/video/{name}/results{name}.json'

    with open(FILE_PATH, "r") as json_file:
        json_data = json.load(json_file)
        with open(f'../output/json/final_data/results{name}.json', 'w') as save_file:
            json.dump(json_data, save_file, ensure_ascii=False, indent="\t")
