# violence_index = [1, 2, 6, 30, 100, 120, 160]
# violence_index.sort()
#
# print(violence_index)
#
# start_frame = violence_index[0]
# end_frame = -1
# tmp = [-1, -1]
# time_pairs = []
#
# one_video_frame = 30  # 30fps 가정
# frame_size = 220
#
#
# def make_pair(start_frame, end_frame):
#     start_video_frame = max(start_frame - 30, 0)
#     end_video_frame = min(frame_size, start_frame + 30 if end_frame == -1 else end_frame + 30)
#     return [start_video_frame, end_video_frame]
#
# def make_time_pair():
#     for cand_idx in violence_index:
#         if end_frame == -1:
#             end_frame = start_frame
#
#         if end_frame + one_video_frame < cand_idx:
#             time_pair = make_pair(start_frame, end_frame)
#             time_pairs.append(time_pair)
#             start_frame = cand_idx
#             end_frame = -1
#         else:
#             end_frame = cand_idx
#
#     # start만 있고 end는 없을 경우
#     if start_frame != end_frame:
#         time_pair = make_pair(start_frame, end_frame)
#         time_pairs.append(time_pair)
#
# print(time_pairs)

import cv2

file_name = 110
OUTPUT_VIDEO_PATH = f"../output/video/{file_name}.avi"
time_list = [[[1, 2], [11, 12]]]

cap = cv2.VideoCapture(OUTPUT_VIDEO_PATH)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = 30
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

for lst in time_list:
    for i in range(lst[0], lst[1] + 1):
        FRAME_PATH = f"../output/video/frames/{i}.png"
        frame_img = cv2.imread(frame_path)
        out.write(frame_img)

cap.release()
out.release()
