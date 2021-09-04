import csv
import os
import cv2
import json
import matplotlib.pyplot as plt
from numba import cuda

from global_data import *
from child_distinguish import child_distinguish
from tracking import tracking
from skeleton import detect_skeleton
from plotting import get_variance


# 전체 프레임에서 bbox만큼 잘라 관절 추출(각도, 기울기는 상대적인 값이기 때문)
# 팔이 범위를 벗어날 수 있기 때문에 가로 2배
def get_skeleton(line, image, frame_id):
    line = list(map(float, line))
    line = list(map(round, line))

    # 이미지 잘라서 저장
    x = line[0] - line[2] // 2
    y = line[1]
    w = line[2] * 2
    h = line[3]
    if x + w > image.shape[1]:
        x1 = image.shape[1]
    else:
        x1 = x + w

    if y + h > image.shape[0]:
        y1 = image.shape[0]
    else:
        y1 = y + h

    if x < 0:
        x = 0

    cropped_image = image[y: y1, x: x1]
    cv2.imwrite(f"../output/video/{get_video_name()}/cropped_image/{frame_id}.png", cropped_image)

    # 잘린 이미지에 대해서 skeleton 뽑아냄
    json_file = detect_skeleton(file_name,
                                ["--image_path", f"../output/video/{get_video_name()}/cropped_image/{frame_id}.png"],
                                'photo', frame_id, True)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def get_first_frame(filename, frame_num):
    path = f'../output/video/{filename}/'
    cap = cv2.VideoCapture(f'../media/{filename}.mp4')

    # frame_num만큼 동영상을 넘김
    for i in range(0, frame_num):
        ret, image = cap.read()

    ret, image = cap.read()

    if ret:
        image = cv2.resize(image, get_frame_size(), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(path + 'frame.png', image)

    cap.release()

    return path + 'frame.png'


def find_adult(file_name, csv_file, frame_num):
    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    rdr = list(rdr)

    skeleton_id = -1
    while skeleton_id == -1:
        # 첫 번째 프레임
        first_frame_image = get_first_frame(file_name, frame_num)

        # 관절로 성인을 찾고 머리가 가장 비슷한 bbox 찾기
        first_frame_json = detect_skeleton(file_name, ["--image_path", first_frame_image], 'photo', frame_num)
        skeleton_id = child_distinguish(first_frame_json, 0)

        if skeleton_id != -1 or int(rdr[-1][0]) < frame_num + 1:
            break

        frame_num += 1

    with open(first_frame_json, 'r') as f:
        json_obj = json.load(f)
    head = json_obj[0]['person'][skeleton_id]['keypoint']['Head']

    minimum = -1  # 가장 작은 차이값
    adult_id = 0  # 차이값이 가장 작은 사람의 id
    idx = 0
    # frame 찾음
    while int(rdr[idx][0]) < frame_num + 1:
        idx += 1
    # frame_id가 동일한 line들을 list로 만듦
    tmp_list = []
    for line in rdr[idx:]:
        if int(line[0]) == frame_num + 1:
            tmp_list.append(line)
            idx += 1
        else:  # 다음 프레임으로 넘어가버렸다!
            break

    for line in tmp_list:
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
    return adult_id, frame_num


if __name__ == "__main__":
    file_name = "800"
    set_video_name(file_name)
    path = f'../media/{file_name}.mp4'

    init_rate()
    # frame_size 정해줌
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    # cropped_image 폴더를 만들어둠
    if not os.path.exists(f"../output/video/{get_video_name()}/cropped_image"):
        os.makedirs(f"../output/video/{get_video_name()}/cropped_image")

    # # MOT 돌리기
    # csv_file = tracking(['mot', '--load_model', '../../FairMOT/models/fairmot_dla34.pth', \
    #                      '--input-video', path, '--input-video-name', file_name, \
    #                      '--output-root', f'../output/video/{file_name}/final', '--conf_thres', '0.4'])
    #
    # # GPU memory 초기화
    # cuda.close()

    csv_file = f'../output/video/{file_name}/final/results{file_name}_0.csv'

    # 성인의 id와 현재 frame 번호
    adult_id, frame_num = find_adult(file_name, csv_file, 0)
    adult_frame = frame_num     # 성인이 탐지된 frame 번호

    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    rdr = list(rdr)

    idx = 0  # 현재 프레임의 마지막줄
    rdr_size = len(rdr)
    skeleton_list = []
    while idx < rdr_size:
        # frame 찾음
        while int(rdr[idx][0]) < frame_num + 1:
            idx += 1

        # frame_id가 동일한 line들을 list로 만듦
        tmp_list = []
        for line in rdr[idx:]:
            if int(line[0]) == frame_num + 1:
                tmp_list.append(line)
                idx += 1
            else:  # 다음 프레임으로 넘어가버렸다!
                break

        image = cv2.imread(f"../output/video/{file_name}/frames/{frame_num}.png")

        not_detected = True
        for item in tmp_list:
            if item[1] == adult_id:
                # 추적 대상 tracking하며 관절 추출
                skeleton_list.extend(get_skeleton(item[2:6], image, frame_num))

                not_detected = False

        # 현재 frame_num에서 성인이 발견되지 않았다면 성인 재탐지
        if not_detected:
            adult_id, frame_num = find_adult(file_name, csv_file, frame_num)
            continue

        frame_num += 1

    # json파일로 저장
    skeleton_json_file = f'../output/video/{file_name}/results{file_name}.json'
    with open(skeleton_json_file, 'w', encoding="utf-8") as make_file:
        json.dump(skeleton_list, make_file, ensure_ascii=False, indent="\t")

    # 관절들의 변화량을 list로 저장
    angle_arm = []; incli_arm = []
    angle_leg = []; incli_leg = []
    for i in range(4):
        angle, incli = get_variance(skeleton_json_file, i)
        if i is 0 or i is 1:
            angle_arm.extend(angle)
            incli_arm.extend(incli)
        elif i is 2 or i is 3:
            angle_leg.extend(angle)
            incli_leg.extend(incli)

    # plotting
    plt.scatter(incli_leg, angle_leg, label="leg")
    plt.scatter(incli_arm, angle_arm, c='red', label="arm")
    plt.xlabel('inclination variance')
    plt.ylabel('angle variance')
    plt.legend()
    plt.show()
