import json
from rw_json import *


def null_exception(cam_num, file_name):
    # json 파일 열기
    try:
        with open(f'../output/json/skeleton_data/violent/cam{cam_num}/output{file_name}.json', 'r') as f:
            json_data = json.load(f)
    except:
        return

    json_len = len(json_data)
    for frame_num in range(0, json_len):
        try:
            neck = json_data[frame_num]['person'][0]['keypoint']['Neck']
        except:
            one_frame_data = {'frame_id': frame_num}
            location = {}
            for keypoint in range(15):
                body = {
                    "x": float(0),
                    "y": float(0),
                    "accuracy": float(0)
                }
                location.update({body_point[keypoint]: body})

            one_frame_data["person"] = [{
                "person_id": 0,
                "keypoint": location
            }]
            json_data.pop(frame_num)
            json_data.insert(frame_num, one_frame_data)

    with open(f'../output/json/skeleton_data/violent/cam{cam_num}/output{file_name}.json', 'w') as f:
        json.dump(json_data, f, indent="\t")


for i in range(1, 3):
    for j in range(1, 116):
        null_exception(str(i), str(j))
