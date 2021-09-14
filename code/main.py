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
    if y < 0:
        y = 0

    cropped_image = image[y: y1, x: x1]
    cv2.imwrite(f"../output/video/{get_video_name()}/cropped_image/{frame_id}.png", cropped_image)

    # 잘린 이미지에 대해서 skeleton 뽑아냄
    json_file = detect_skeleton(get_video_name(),
                                ["--image_path", f"../output/video/{get_video_name()}/cropped_image/{frame_id}.png"],
                                'photo', frame_id, True)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


def find_adult(csv_file, frame_num):
    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    rdr = list(rdr)

    skeleton_id = -1
    total_frame = len(os.listdir(f'../output/video/{get_video_name()}/frames'))
    while skeleton_id == -1:
        # 프레임 수가 넘어가면 break
        if total_frame <= frame_num:
            break

        # 첫 번째 프레임
        first_frame_image = f'../output/video/{get_video_name()}/frames/{frame_num}.png'

        # 관절로 성인을 찾고 머리가 가장 비슷한 bbox 찾기
        first_frame_json = detect_skeleton(get_video_name(), ["--image_path", first_frame_image], 'photo', frame_num)
        skeleton_id = child_distinguish(0, first_frame_json)

        frame_num += 1

    with open(first_frame_json, 'r') as f:
        json_obj = json.load(f)
    head = json_obj[0]['person'][skeleton_id]['keypoint']['Head']
    neck = json_obj[0]['person'][skeleton_id]['keypoint']['Neck']

    # 성인의 bbox를 확인할 수 없음
    if head['accuracy'] < 0.5 or neck['accuracy'] < 0.5:
        return -2, frame_num + 1, first_frame_json, skeleton_id

    minimum = -1  # 가장 작은 차이값
    adult_id = -1  # 차이값이 가장 작은 사람의 id
    idx = 0
    # frame 찾음
    while int(rdr[idx][0]) < frame_num + 1:
        idx += 1
        # bbox를 찾을 수 없음
        if idx >= len(rdr):
            return -2, frame_num + 1, first_frame_json, skeleton_id
    # frame_id가 동일한 line들을 list로 만듦
    tmp_list = []
    for line in rdr[idx:]:
        if int(line[0]) == frame_num + 1:
            tmp_list.append(line)
            idx += 1
        else:  # 다음 프레임으로 넘어가버렸다!
            break

    for line in tmp_list:
        # neck의 좌표가 bbox의 범위 내에 있는지 확인
        if not (float(line[2]) <= neck['x'] <= float(line[2]) + float(line[4]) and \
                float(line[3]) <= neck['y'] <= float(line[3]) + float(line[5])):
            continue

        # x좌표
        x = float(line[2]) + float(line[4]) / 2
        # y좌표, y좌표가 음수인 경우를 대비해 max 함수 사용
        y = max(float(line[3]), 0)

        # x좌표와 y좌표의 차이
        x_diff = abs(head["x"] - x)
        y_diff = head["y"] - y

        # 머리가 bbox보다 위에 있으면 continue
        if y_diff < 0:
            continue

        if minimum == -1:
            minimum = x_diff + y_diff
            adult_id = line[1]
        else:
            if minimum > x_diff + y_diff:
                minimum = x_diff + y_diff
                adult_id = line[1]

    f.close()
    return adult_id, frame_num, -1, -1


def tracking_by_skeleton(json_obj, skeleton_list, frame_num, skeleton_id):
    distance = 0
    key_count = 0
    for key in body_point[:-1]:
        if json_obj[0]['person'][skeleton_id]['keypoint'][key]['accuracy'] >= 0.5:
            x = json_obj[0]['person'][skeleton_id]['keypoint'][key]['x']
            y = json_obj[0]['person'][skeleton_id]['keypoint'][key]['y']
            key_count += 1
        else:
            continue
        if len(skeleton_list) != 0:
            skeleton_id2 = child_distinguish(0, "", [skeleton_list[-1]])
            x2 = skeleton_list[-1]['person'][skeleton_id2]['keypoint'][key]['x']
            y2 = skeleton_list[-1]['person'][skeleton_id2]['keypoint'][key]['y']
        # skeleton_list가 비어있으면(첫 번째 프레임이어서 성인 탐지가 안되어 있으면)
        # 무조건 성인으로 탐지한 skeleton 값을 집어넣음
        else:
            json_obj[0]['person'][skeleton_id]['person_id'] = 0
            skeleton_list.append({"frame_id": frame_num - 1, "person": [json_obj[0]['person'][skeleton_id]]})
            return skeleton_list
        distance += abs(x - x2) + abs(y - y2)

    w, h = get_frame_size()
    if distance / key_count < w / 16:
        json_obj[0]['person'][skeleton_id]['person_id'] = 0
        skeleton_list.append({"frame_id": frame_num - 1, "person": [json_obj[0]['person'][skeleton_id]]})
    return skeleton_list


def crop(skeleton_list):
    i = 0
    while i < len(skeleton_list):
        frame_num = skeleton_list[i]['frame_id']
        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")
        x1 = int(skeleton_list[i]['person'][0]['keypoint']['LShoulder']['x'])
        x2 = int(skeleton_list[i]['person'][0]['keypoint']['RShoulder']['x'])
        y1 = int(skeleton_list[i]['person'][0]['keypoint']['Head']['y'])
        y2 = int(skeleton_list[i]['person'][0]['keypoint']['Chest']['y'])
        cropped_image = image[y1: y2, x1: x2]
        cv2.imwrite(f"../output/video/{get_video_name()}/cropped_image/{frame_num}.png", cropped_image)
        i += 1


if __name__ == "__main__":
    file_name = "218"
    set_video_name(file_name)
    path = f'../media/{get_video_name()}.mp4'

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
    #                      '--input-video', path, '--input-video-name', get_video_name(), \
    #                      '--output-root', f'../output/video/{get_video_name()}/final', '--conf_thres', '0.4'])
    #
    # # GPU memory 초기화
    # cuda.close()

    csv_file = f'../output/video/{get_video_name()}/final/results{get_video_name()}_0.csv'

    # 성인의 관절을 저장한 list
    skeleton_list = []
    # 성인의 id와 현재 frame 번호
    adult_id, frame_num, adult_json, skeleton_id = find_adult(csv_file, 0)

    # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
    if adult_id == -2:
        with open(adult_json, 'r') as f:
            json_obj = json.load(f)
        skeleton_list = tracking_by_skeleton(json_obj, skeleton_list, frame_num, skeleton_id)

    f = open(csv_file, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    rdr = list(rdr)

    idx = 0  # 현재 프레임의 첫번째줄
    rdr_size = len(rdr)
    total_frame = len(os.listdir(f'../output/video/{get_video_name()}/frames'))
    while frame_num < total_frame:
        # frame 찾음
        while idx < len(rdr) and int(rdr[idx][0]) < frame_num + 1:
            idx += 1

        # frame_id가 동일한 line들을 list로 만듦
        tmp_list = []
        for line in rdr[idx:]:
            if int(line[0]) == frame_num + 1:
                tmp_list.append(line)
                idx += 1
            else:  # 다음 프레임으로 넘어가버렸다!
                break

        # 검출된 bbox가 없는 경우
        if len(tmp_list) == 0:
            # 성인의 id를 새로 찾기 위해 -1로 초기화
            adult_id = -1
            adult_id, frame_num, adult_json, skeleton_id = find_adult(csv_file, frame_num)

            # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
            if adult_id == -2:
                with open(adult_json, 'r') as f:
                    json_obj = json.load(f)
                skeleton_list = tracking_by_skeleton(json_obj, skeleton_list, frame_num, skeleton_id)
            continue

        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")

        not_detected = True
        for item in tmp_list:
            if item[1] == adult_id:
                # 추적 대상 tracking하며 관절 추출
                skeleton_list.extend(get_skeleton(item[2:6], image, frame_num))

                not_detected = False

        # 현재 frame_num에서 성인이 발견되지 않았다면 성인 재탐지
        if not_detected:
            # 성인의 id를 새로 찾기 위해 -1로 초기화
            adult_id = -1
            adult_id, frame_num, adult_json, skeleton_id = find_adult(csv_file, frame_num)

            # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
            if adult_id == -2:
                with open(adult_json, 'r') as f:
                    json_obj = json.load(f)
                skeleton_list = tracking_by_skeleton(json_obj, skeleton_list, frame_num, skeleton_id)
            continue

        frame_num += 1

    # json파일로 저장
    skeleton_json_file = f'../output/video/{get_video_name()}/results{get_video_name()}.json'
    with open(skeleton_json_file, 'w', encoding="utf-8") as make_file:
        json.dump(skeleton_list, make_file, ensure_ascii=False, indent="\t")

    # crop(skeleton_list)

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
