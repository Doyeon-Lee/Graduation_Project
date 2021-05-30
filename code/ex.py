# import json
# frame_data = []
#
# for frame_id in range(3):
#     keypoints = {"frame_id": frame_id}
#     keypoints_tmp = []
#     for person_id in range(2):
#         location = {}
#         body_point = []
#         for keypoint in range(5):
#             body_point.append({"x": float(1),
#                                "y": float(2),
#                                "accuracy": float(3)})
#         location.update({'body_point': body_point})
#
#         keypoints_tmp.append({
#                 "person_id": person_id,
#                 "keypoint": location
#         })
#     keypoints["person"] = keypoints_tmp
#     frame_data.append(keypoints)
#
#     with open('../output/output.json', 'w', encoding="utf-8") as make_file:
#         json.dump(frame_data, make_file, ensure_ascii=False, indent="\t")

import numpy as np


def get_avg(val_list):
    num_avg = 5
    tmp = []
    s = len(val_list)
    iter = s // num_avg

    for i in range(iter):
        val = i*num_avg
        tmp.append(np.sum(val_list[val:val + num_avg]) / num_avg)

    if s % num_avg > 0:
        tmp.extend(val_list[num_avg * iter:])
    return tmp


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
print(get_avg(a))