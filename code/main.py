from skeleton import *
from video_tracking import *

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



# json파일 첫 frame에서 bbox 찾고 동영상에 tracking 적용하기
with open('../output/output.json') as f:
    json_data = json.load(f)

    obj = json_data[0]['person'][0]['keypoint']  # 0번째 frame 0번째 person

    tmp = []
    for key in body_point:
        if key == 'Background': break
        tmp.append(obj[key])

    dict_to_list = []
    for dict in tmp:
        tmp2 = []
        for k, v in dict.items():
            tmp2.append(v)
        dict_to_list.append(tmp2)

    VIDEO_PATH = '../output/output.avi'
    cap = cv2.VideoCapture(VIDEO_PATH)
    if cap.isOpened():
        ret, frame = cap.read()
        set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        x, y, w, h = find_bbox2(dict_to_list)  # check_boundary(dict_to_list, FRAME_W, FRAME_H)
        print(x, y, w, h)
        obj_tracking(x, y, w, h)


#frame_data = detect_skeleton()
#get_location(frame_data)


# def angle_between(p1, p2, p3):
#     x1, y1 = p1
#     x2, y2 = p2
#     x3, y3 = p3
#     deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
#     deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
#     return abs(deg2 - deg1)
#
#
# # plotting
# with open('../output/output.json') as f:
#     json_data = json.load(f)
#     json_len = len(json_data)
#
#     right_arm_angle = []
#     for i in range(json_len):
#         obj = json_data[i]['person'][0]['keypoint']  # 0번째 frame 0번째 person
#         rs = obj['RShoulder']
#         re = obj['RElbow']
#         rw = obj['RWrist']
#
#         angle = angle_between((rs['x'], rs['y']), (re['x'], re['y']), (rw['x'], rw['y']))
#         right_arm_angle.append(angle)
#
#     xspan = np.array([i for i in range(json_len)])
#     plt.plot(xspan, right_arm_angle)
#     plt.xlabel('frame')
#     plt.ylabel('right arm angle')
#     plt.show()


