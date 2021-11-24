import csv
import os
import cv2
import json
import sys
from numba import cuda

from global_data import *
from child_distinguish import child_distinguish
from tracking import tracking
from skeleton import detect_skeleton
from plotting import get_distance
from clustering import clustering


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

    x = max(x, 0)
    y = max(y, 0)

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
        # 프레임 수가 넘어가면 return
        if total_frame <= frame_num:
            return -1, total_frame, None, -1

        # 첫 번째 프레임
        first_frame_image = f'../output/video/{get_video_name()}/frames/{frame_num}.png'

        # 관절로 성인을 찾고 머리가 가장 비슷한 bbox 찾기
        first_frame_json = detect_skeleton(get_video_name(), ["--image_path", first_frame_image], 'photo', frame_num)
        skeleton_id = child_distinguish(frame_num, first_frame_json)

        if skeleton_id != -1:
            break

        frame_num += 1

    with open(first_frame_json, 'r') as f:
        json_obj = json.load(f)
    f.close()
    head = json_obj[0]['person'][skeleton_id]['keypoint']['Head']
    neck = json_obj[0]['person'][skeleton_id]['keypoint']['Neck']

    # 성인의 bbox를 확인할 수 없음
    if head['accuracy'] < 0.7 or neck['accuracy'] < 0.7:
        return -2, frame_num, first_frame_json, skeleton_id

    minimum = -1  # 가장 작은 차이값
    adult_id = -2  # 차이값이 가장 작은 사람의 id
    idx = 0
    # frame 찾음
    while int(rdr[idx][0]) < frame_num + 1:
        idx += 1
        # bbox를 찾을 수 없음
        if idx >= len(rdr):
            return -2, frame_num, first_frame_json, skeleton_id
    # frame_id가 동일한 line들을 list로 만듦
    tmp_list = []
    rdr_list = rdr[idx:]
    for line in rdr_list:
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
        y_diff = head["y"] + 1 - y  # 1은 bumper

        # 머리가 bbox보다 위에 있으면 continue
        if y_diff < 0:
            continue

        if minimum == -1:
            minimum = x_diff + y_diff
            adult_id = int(line[1])
        else:
            if minimum > x_diff + y_diff:
                minimum = x_diff + y_diff
                adult_id = int(line[1])

    # 찾은 성인이 이전에 skeleton으로 추적하던 성인과 동일한지 확인
    skeleton_list = get_skeleton_list()
    if len(skeleton_list) > 0:
        distance, key_count = get_distance(json_obj, skeleton_id)
        w, h = get_frame_size()
        skipped_frame_num = frame_num - skeleton_list[-1]['frame_id']  # 현재 프레임과의 차이
        max_range = ((get_prev_adult_head_len()**2 * 56) / (165 * h)) * skipped_frame_num
        if key_count > 0 and distance / key_count < max_range:
            set_prev_adult_point(get_current_adult_point())
        else:
            adult_id = -2
    # skeleton_list가 비어있고, 처음부터 bbox로 탐지를 시작했으면
    # prev_adult_point가 비어있으므로 current_adult_point로 set해줌
    else:
        set_prev_adult_point(get_current_adult_point())
    return adult_id, frame_num, first_frame_json, skeleton_id


def tracking_by_skeleton(json_obj, frame_num, skeleton_id):
    skeleton_list = get_skeleton_list()
    # skeleton_list가 비어있으면(첫 번째 프레임이어서 성인 탐지가 안되어 있으면)
    # 무조건 성인으로 탐지한 skeleton 값을 집어넣음
    if len(skeleton_list) == 0:
        set_prev_adult_point(get_current_adult_point())

        json_obj[0]['person'][skeleton_id]['person_id'] = 0
        append_skeleton_list({"frame_id": frame_num, "person": [json_obj[0]['person'][skeleton_id]]})
        crop([skeleton_list[-1]])
        return

    distance, key_count = get_distance(json_obj, skeleton_id)

    w, h = get_frame_size()
    skipped_frame_num = frame_num - skeleton_list[-1]['frame_id']   # 현재 프레임과의 차이
    max_range = ((get_prev_adult_head_len()**2 * 56) / (165 * h)) * skipped_frame_num
    if key_count > 0 and distance / key_count < max_range:
        set_prev_adult_point(get_current_adult_point())

        json_obj[0]['person'][skeleton_id]['person_id'] = 0
        append_skeleton_list({"frame_id": frame_num, "person": [json_obj[0]['person'][skeleton_id]]})
        crop([skeleton_list[-1]])
    return


def crop(skeleton_list):
    i = 0
    while i < len(skeleton_list):
        frame_num = skeleton_list[i]['frame_id']
        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")
        x1 = int(skeleton_list[i]['person'][0]['keypoint']['LShoulder']['x'])
        x2 = int(skeleton_list[i]['person'][0]['keypoint']['RShoulder']['x'])
        y1 = int(skeleton_list[i]['person'][0]['keypoint']['Head']['y'])
        y2 = int(skeleton_list[i]['person'][0]['keypoint']['Chest']['y'])

        if x1 > x2:
            x1, x2 = swap(x1, x2)
        if y1 > y2:
            y1, y2 = swap(y1, y1)

        cropped_image = image[y1: y2, x1: x2]
        cv2.imwrite(f"../output/video/{get_video_name()}/cropped_image/{frame_num}.png", cropped_image)
        i += 1


def exec_MOT(original_video_path):
    # cropped_image 폴더를 만들어둠
    CROPPED_IMAGE_PATH = f"../output/video/{get_video_name()}/cropped_image"
    if not os.path.exists(CROPPED_IMAGE_PATH):
        os.makedirs(CROPPED_IMAGE_PATH)

    # 최종 결과물이 들어갈 폴더를 만들어줌
    FINAL_RESULT_PATH = f"../output/final_results"
    if not os.path.exists(FINAL_RESULT_PATH):
        os.makedirs(FINAL_RESULT_PATH)

    # MOT 돌리기
    MODEL_PATH = '../../FairMOT/models/fairmot_dla34.pth'
    OUTPUT_VIDEO_PATH = f'../output/video/{get_video_name()}/final'
    csv_file = tracking(['mot', '--load_model', MODEL_PATH, '--input-video', original_video_path, \
                         '--input-video-name', get_video_name(), '--output-root', OUTPUT_VIDEO_PATH, '--conf_thres', '0.4'])

    # GPU memory 초기화
    cuda.close()


# 첫번째 프레임의 adult id, frame번호 등을 처리한다
def handle_first_frame_info(BBOX_CSV_PATH):
    adult_id = -2
    frame_num = 0
    while adult_id == -2:
        # 성인의 id와 현재 frame 번호
        adult_id, frame_num, adult_json, skeleton_id = find_adult(BBOX_CSV_PATH, frame_num)

        # frame_num이 total_frame을 넘었음(영상이 끝남)
        if adult_id == -1:
            break
        # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
        if adult_id == -2:
            with open(adult_json, 'r') as f:
                json_obj = json.load(f)
            tracking_by_skeleton(json_obj, frame_num, skeleton_id)
            frame_num += 1
    return adult_id, frame_num


def get_first_adult_id(BBOX_CSV_PATH):
    saved_adult_id = -2
    adult_id, frame_num = handle_first_frame_info(BBOX_CSV_PATH)

    # 어른으로 인식된 객체가 있을 때
    if adult_id >= 0:
        saved_adult_id = adult_id

    csv_file = open(BBOX_CSV_PATH, 'r', encoding='utf-8')
    csv_rdr = csv.reader(csv_file)
    csv_list = list(csv_rdr)
    csv_idx = 0  # 현재 프레임의 첫번째줄
    is_frame_passed = True  # 프레임이 넘어갔는지 여부
    csv_size = len(csv_list)
    total_frame = len(os.listdir(f'../output/video/{get_video_name()}/frames'))

    while frame_num < total_frame:
        # frame 찾음
        while csv_idx < csv_size and int(csv_list[csv_idx][0]) < frame_num + 1:
            csv_idx += 1

        tmp_list = []
        # frame_id가 동일한 line들을 list로 만듦
        if is_frame_passed:
            # 다음 프레임으로 넘어가기 전까지 반복
            while int(csv_list[csv_idx][0]) == frame_num + 1:
                if csv_idx >= csv_size: break
                tmp_list.append(csv_list[csv_idx])
                csv_idx += 1

        # 검출된 bbox가 없는 경우
        if len(tmp_list) == 0:
            before_frame_num = frame_num
            adult_id, frame_num, adult_json, skeleton_id = find_adult(BBOX_CSV_PATH, frame_num)
            # find_adult 함수 내부에서 frame_num이 넘어갔는지 확인
            is_frame_passed = (before_frame_num != frame_num)

            # frame_num이 total_frame을 넘었음(영상이 끝남)
            if adult_id == -1:
                break

            # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
            if adult_id == -2:
                with open(adult_json, 'r') as f:
                    json_obj = json.load(f)
                tracking_by_skeleton(json_obj, frame_num, skeleton_id)
                frame_num += 1
                is_frame_passed = True
            elif adult_id >= 0:
                saved_adult_id = adult_id
            continue

        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")
        not_detected = True

        for item in tmp_list:
            # 추적 대상 tracking하며 관절 추출
            if int(item[1]) == saved_adult_id:
                extend_skeleton_list(get_skeleton(item[2:6], image, frame_num))
                not_detected = False
                # break 넣어도 되지 않나?

        # 현재 frame_num에서 성인이 발견되지 않았다면 성인 재탐지
        if not_detected:
            before_frame_num = frame_num
            adult_id, frame_num, adult_json, skeleton_id = find_adult(BBOX_CSV_PATH, frame_num)
            # find_adult 함수 내부에서 frame_num이 넘어갔는지 확인
            is_frame_passed = (before_frame_num != frame_num)

            # frame_num이 total_frame을 넘었음(영상이 끝남)
            if adult_id == -1:
                break

            # skeleton 추출은 되지만 bbox가 없는 경우 skeleton 자체를 skeleton_list에 append 해줌
            if adult_id == -2:
                with open(adult_json, 'r') as f:
                    json_obj = json.load(f)
                tracking_by_skeleton(json_obj, frame_num, skeleton_id)
                frame_num += 1
                is_frame_passed = True
            elif adult_id >= 0:
                saved_adult_id = adult_id
            continue

        frame_num += 1
        is_frame_passed = True

    return adult_id


def make_violence_video(SKELETON_JSON_PATH):
    time_list = clustering(SKELETON_JSON_PATH)
    pathOut = f'../output/final_results/{get_video_name()}.avi'
    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), 30, get_frame_size())
    for lst in time_list:
        for i in range(lst[0], lst[1] + 1):
            pathIn = f'../output/video/{get_video_name()}/frames/{i}.png'
            img = cv2.imread(pathIn)
            out.write(img)
    out.release()


if __name__ == "__main__":
    original_video_name = sys.argv[1]
    if sys.argv[2] == "n":
        original_video_path = f'../media/non-violence/{original_video_name}.mp4'
    else:
        original_video_path = f'../media/violence/{original_video_name}.mp4'

    set_video_name(original_video_name)
    init_rate()
    init_point()

    # frame_size 정해줌
    cap = cv2.VideoCapture(original_video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    set_frame_size(int(width), int(height))
    cap.release()

    exec_MOT(original_video_path)

    BBOX_CSV_PATH = f'../output/video/{get_video_name()}/final/results{get_video_name()}_0.csv'
    adult_id = get_first_adult_id(BBOX_CSV_PATH)

    # json파일로 저장
    SKELETON_JSON_PATH = f'../output/video/{get_video_name()}/results{get_video_name()}.json'
    with open(SKELETON_JSON_PATH, 'w', encoding="utf-8") as make_file:
        json.dump(get_skeleton_list(), make_file, ensure_ascii=False, indent="\t")

    # 폭력 의심 구간을 영상으로 만듦
    make_violence_video(SKELETON_JSON_PATH)
