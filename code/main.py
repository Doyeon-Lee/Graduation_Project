import cv2
from video_example import *
import json


body_point = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
              "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
              "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
              "REye", "LEye", "REar", "LEar", "LBigToe",
              "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
# body_point = ['Neck', 'LShoulder', 'RShoulder', 'MidHip', 'LHip', 'RHip']
body_dict = {}
for i, v in enumerate(body_point):
    body_dict[v] = i


# def check_boundary(person_bodypoint, frame_w, frame_h):
#     n = body_dict['Neck']
#     ls = body_dict['LShoulder']
#     rs = body_dict['RShoulder']
#     mh = body_dict['MidHip']
#     lh = body_dict['LHip']
#     rh = body_dict['RHip']
#
#     shoulder_len = max(person_bodypoint[n][0] - person_bodypoint[rs][0],
#                        person_bodypoint[n][0] - person_bodypoint[ls][0])
#
#     _x = max(0, int(person_bodypoint[n][0] - shoulder_len))
#     _y = int(person_bodypoint[n][1])
#     _w = 0
#     _h = 0
#
#     # MidHip이 안보일 경우
#     if not person_bodypoint[mh][1]:
#         _h = int(frame_h - person_bodypoint[n][1])
#     else:
#         _h = int(person_bodypoint[mh][1] - person_bodypoint[n][1])
#
#     # Neck == 0
#     if not person_bodypoint[n][0]:
#         if not person_bodypoint[lh][0]:  # LHip == 0
#             _w = min(person_bodypoint[rh][0], FRAME_W - person_bodypoint[rh][0])
#         elif not person_bodypoint[rh][0]:
#             _w = min(person_bodypoint[lh][0], FRAME_W - person_bodypoint[lh][0])
#         else:
#             _w = abs(person_bodypoint[rh][0] - person_bodypoint[lh][0])
#
#     else:  # Neck != 0
#         if not person_bodypoint[mh][0]:  # MidHip == 0
#             if not person_bodypoint[ls][0]:  # LShoulder == 0
#                 _w = min(person_bodypoint[rs][0], FRAME_W - person_bodypoint[rs][0])
#             elif not person_bodypoint[rs][0]:  # RShoulder == 0
#                 _w = min(person_bodypoint[ls][0], FRAME_W - person_bodypoint[ls][0])
#         else:  # Neck != 0 and MidHip != 0
#             _w = abs(person_bodypoint[ls][0] - person_bodypoint[rs][0])
#
#     _w = int(_w)
#     return _x, _y, _w, _h
#
#
# # load json
with open('../output/output.json') as f:
    json_data = json.load(f)

    obj = json_data[0]['person'][0]['keypoint'] # 0번째 frame 0번째 person

    tmp = []
    for key in body_point:
        tmp.append(obj[key])

    dict_to_list = []
    for dict in tmp:
        tmp2 = []
        for k, v in dict.items():
            tmp2.append(v)
        dict_to_list.append(tmp2)

    VIDEO_PATH = '../media/17.mp4'
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        ret, frame = cap.read()
        FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x, y, w, h = check_boundary(dict_to_list, FRAME_W, FRAME_H)
        print(x, y, w, h)
        obj_tracking(x, y, w, h)


#frame_data = detect_skeleton()
#get_location(frame_data)
