import csv

from child_distinguish import *
from tracking import tracking
from skeleton import *
from plotting import *


# 전체 프레임에서 bbox만큼 잘라 관절 추출(각도, 기울기는 상대적인 값이기 때문)
# 팔이 범위를 벗어날 수 있기 때문에 가로 2배
def get_skeleton(line, image, frame_id):
    line = list(map(float, line))
    line = list(map(round, line))

    # 이미지 잘라서 저장
    x = line[0] - line[2] // 2
    if x < 0:
        x = 0

    w = line[2] * 2
    if x + w > image.shape[1]:
        x1 = image.shape[1]
    else:
        x1 = x + w

    cropped_image = image[line[1]: line[1] + line[3], x: x1]
    cv2.imwrite("cropped_image.png", cropped_image)

    # 잘린 이미지에 대해서 skeleton 뽑아냄
    json_file = detect_skeleton(file_name, ["--image_path", "cropped_image.png"], 'photo', frame_id)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


# def get_first_frame(filename, frame_num):
#     path = f'../output/video/{filename}/'
#     cap = cv2.VideoCapture(f'../media/{filename}.mp4')
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#     set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(path + f'{filename}_0.avi', fourcc, 1, get_frame_size())
#
#     # frame_num만큼 동영상을 넘김
#     for i in range(0, frame_num):
#         ret, image = cap.read()
#
#     ret, image = cap.read()
#     if ret:
#         out.write(image)
#
#     return path + f'{filename}_0.avi'

def get_first_frame(filename, frame_num):
    path = f'../output/video/{filename}/'
    cap = cv2.VideoCapture(f'../media/{filename}.mp4')

    # frame_num만큼 동영상을 넘김
    for i in range(0, frame_num):
        ret, image = cap.read()

    ret, image = cap.read()
    if ret:
        cv2.imwrite(path + f'frame{frame_num}.png', image)

    cap.release()

    return path + f'frame{frame_num}.png'


if __name__ == "__main__":
    file_name = "54"
    path = f'../media/{file_name}.mp4'

    skeleton_id = -1
    frame_num = 0
    while skeleton_id == -1:
        # 첫 번째 프레임
        first_frame_video = get_first_frame(file_name, frame_num)

        # 관절로 성인을 찾고 머리가 가장 비슷한 bbox 찾기
        first_frame_json = detect_skeleton(file_name, ["--image_path", first_frame_video], 'photo', frame_num)
        skeleton_id = child_distinguish(first_frame_json, frame_num)

        if skeleton_id != -1:
            break

        frame_num += 1

    # MOT 돌리기
    # csv_file = tracking(['mot', '--load_model', '../../FairMOT/models/fairmot_dla34.pth', \
    #                      '--input-video', path, '--input-video-name', file_name, \
    #                      '--output-root', f'../output/video/{file_name}/final', '--conf_thres', '0.4'])

    csv_file = f'../output/video/{file_name}/final/results54_0.csv'
    with open(first_frame_json, 'r') as f:
        json_obj = json.load(f)
    head = json_obj[frame_num]['person'][skeleton_id]['keypoint']['Head']

    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)

    minimum = -1  # 가장 작은 차이값
    adult_id = 0  # 차이값이 가장 작은 사람의 id
    for line in rdr:
        # frame 찾음
        if int(line[0]) < frame_num + 1:
            continue

        # x좌표
        x = float(line[2]) + float(line[4]) / 2
        # y좌표
        y = float(line[3])

        # x좌표와 y좌표의 차이
        x_diff = abs(head["x"] - x)
        y_diff = abs(head["y"] - y)

        if minimum == -1:
            minimum = x_diff + y_diff
            adult_id = line[1]
        else:
            if minimum > x_diff + y_diff:
                minimum = x_diff + y_diff
                adult_id = line[1]

    f.close()

    cap = cv2.VideoCapture(path)
    # frame_num만큼 동영상을 넘김
    for i in range(0, frame_num):
        ret, image = cap.read()

    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    skeleton_list = []
    for line in rdr:
        # frame 찾음
        if int(line[0]) < frame_num + 1:
            continue
        while int(line[0]) > frame_num + 1:
            frame_num += 1
            ret, image = cap.read()

        # 성인 찾음
        if line[1] != adult_id:
            continue

        # image capture
        ret, image = cap.read()

        # 추적 대상 tracking하며 관절 추출
        skeleton_list.extend(get_skeleton(line[2:6], image, frame_num))

        frame_num += 1

    with open(first_frame_json, 'w', encoding="utf-8") as make_file:
        json.dump(skeleton_list, make_file, ensure_ascii=False, indent="\t")

    cap.release()

# while(동영상이 끝날때까지):
#
# 	get_skeleton(x, y, w, h)
#
# 	if(id가 달라짐):
# 		# 어린이집 cctv의 경우
# 		adult_id = find_adult(file_name, frame_num)
