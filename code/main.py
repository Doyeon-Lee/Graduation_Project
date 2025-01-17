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


def save_cropped_image_bbox(csv_line, image, frame_id):
    csv_line = list(map(float, csv_line))
    csv_line = list(map(round, csv_line))

    # 이미지 잘라서 저장
    x = csv_line[0] - csv_line[2] // 2
    y = csv_line[1]
    w = csv_line[2] * 2
    h = csv_line[3]

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


# 전체 프레임에서 bbox만큼 잘라 관절 추출(각도, 기울기는 상대적인 값이기 때문)
# 팔이 범위를 벗어날 수 있기 때문에 가로 2배
def get_skeleton_json(frame_id):
    # 잘린 이미지에 대해서 skeleton 뽑아냄
    json_file = detect_skeleton(get_video_name(),
                                ["--image_path", f"../output/video/{get_video_name()}/cropped_image/{frame_id}.png"],
                                'photo', frame_id, True)
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data


# frame_num 부터 성인을 찾아 다양한 정보를 리턴
def find_adult(BBOX_CSV_PATH, frame_num):
    csv_file = open(BBOX_CSV_PATH, 'r', encoding='utf-8')
    csv_rdr = csv.reader(csv_file)
    csv_list = list(csv_rdr)
    skeleton_id = -1
    total_frame = len(os.listdir(f'../output/video/{get_video_name()}/frames'))

    while skeleton_id == -1:
        # 영상이 끝날 때까지 찾지 못했음
        if total_frame <= frame_num:
            return -1, total_frame, None, -1

        # 첫 번째 프레임
        frame_image = f'../output/video/{get_video_name()}/frames/{frame_num}.png'
        # 관절로 성인을 찾고 머리가 가장 비슷한 bbox 찾기
        frame_json_path = detect_skeleton(get_video_name(), ["--image_path", frame_image], 'photo', frame_num)
        skeleton_id = child_distinguish(frame_num, frame_json_path)

        # while 문의 조건이 있지만 아래의 frame_num + 1을 방지하기 위해 미리 break 해준다
        if skeleton_id != -1: break
        frame_num += 1

    # frame_json_path 가 정의되지 않는 경우는 위 if문에서 return되기 때문에 상관없다
    with open(frame_json_path, 'r') as f:
        json_obj = json.load(f)
        head = json_obj[0]['person'][skeleton_id]['keypoint']['Head']
        neck = json_obj[0]['person'][skeleton_id]['keypoint']['Neck']

    # 관절값을 믿을 수 없음
    if head['accuracy'] < 0.7 or neck['accuracy'] < 0.7:
        return -2, frame_num, frame_json_path, skeleton_id

    csv_size = len(csv_list)
    csv_tmp_list = []
    csv_idx = 0
    min_sum = -1  # 가장 작은 차이값
    adult_id = -2  # 차이값이 가장 작은 사람의 id

    # frame 찾음
    while int(csv_list[csv_idx][0]) < frame_num + 1:
        csv_idx += 1
        # bbox를 찾을 수 없음
        if csv_idx >= len(csv_list):
            return -2, frame_num, frame_json_path, skeleton_id

        # frame_id가 동일한 line들을 list로 만듦
        while int(csv_list[csv_idx][0]) == frame_num + 1:
            if csv_idx >= csv_size: break
            csv_tmp_list.append(csv_list[csv_idx])
            csv_idx += 1

    for csv_tmp_line in csv_tmp_list:
        # neck의 좌표가 bbox의 범위 내에 있는지 확인
        if not (float(csv_tmp_line[2]) <= neck['x'] <= float(csv_tmp_line[2]) + float(csv_tmp_line[4]) and \
                float(csv_tmp_line[3]) <= neck['y'] <= float(csv_tmp_line[3]) + float(csv_tmp_line[5])):
            continue

        # x좌표
        bbox_x = float(csv_tmp_line[2]) + float(csv_tmp_line[4]) / 2
        # y좌표, y좌표가 음수인 경우를 대비해 max 함수 사용
        bbox_y = max(float(csv_tmp_line[3]), 0)

        # x좌표와 y좌표의 차이
        x_diff = abs(head["x"] - bbox_x)
        y_diff = head["y"] + 1 - bbox_y  # 1은 bumper

        # 머리가 bbox보다 위에 있으면 continue
        if y_diff < 0:
            continue

        if min_sum == -1:
            min_sum = x_diff + y_diff
            adult_id = int(csv_tmp_line[1])
        else:
            if min_sum > x_diff + y_diff:
                min_sum = x_diff + y_diff
                adult_id = int(csv_tmp_line[1])

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
    return adult_id, frame_num, frame_json_path, skeleton_id


def track_by_skeleton(json_obj, frame_num, skeleton_id):
    # skeleton_list가 비어있으면(첫 번째 프레임이어서 성인 탐지가 안되어 있으면)
    # 무조건 성인으로 탐지한 skeleton 값을 집어넣음
    skeleton_list = get_skeleton_list()
    if len(skeleton_list) == 0:
        set_prev_adult_point(get_current_adult_point())
        json_obj[0]['person'][skeleton_id]['person_id'] = 0
        append_skeleton_list({"frame_id": frame_num, "person": [json_obj[0]['person'][skeleton_id]]})
        save_cropped_image_skeleton([skeleton_list[-1]])
        return

    distance, key_count = get_distance(json_obj, skeleton_id)
    w, h = get_frame_size()
    skipped_frame_num = frame_num - skeleton_list[-1]['frame_id']   # 현재 프레임과의 차이
    max_range = ((get_prev_adult_head_len()**2 * 56) / (165 * h)) * skipped_frame_num

    if key_count > 0 and distance / key_count < max_range:
        set_prev_adult_point(get_current_adult_point())
        json_obj[0]['person'][skeleton_id]['person_id'] = 0
        append_skeleton_list({"frame_id": frame_num, "person": [json_obj[0]['person'][skeleton_id]]})
        save_cropped_image_skeleton([skeleton_list[-1]])
    return


def save_cropped_image_skeleton(skeleton_list):
    i = 0
    while i < len(skeleton_list):
        frame_num = skeleton_list[i]['frame_id']
        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")
        tmp_keypoint = skeleton_list[i]['person'][0]['keypoint']
        x1 = int(tmp_keypoint['LShoulder']['x'])
        x2 = int(tmp_keypoint['RShoulder']['x'])
        y1 = int(tmp_keypoint['Head']['y'])
        y2 = int(tmp_keypoint['Chest']['y'])

        if x1 > x2:
            x1, x2 = swap(x1, x2)
        if y1 > y2:
            y1, y2 = swap(y1, y1)

        cropped_image = image[y1: y2, x1: x2]
        CROPPED_IMAGE_PATH = f"../output/video/{get_video_name()}/cropped_image/{frame_num}.png"
        cv2.imwrite(CROPPED_IMAGE_PATH, cropped_image)
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
            track_by_skeleton(json_obj, frame_num, skeleton_id)
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

        csv_tmp_list = []
        # frame_id가 동일한 line들을 list로 만듦
        if is_frame_passed:
            # 다음 프레임으로 넘어가기 전까지 반복
            while int(csv_list[csv_idx][0]) == frame_num + 1:
                if csv_idx >= csv_size: break
                csv_tmp_list.append(csv_list[csv_idx])
                csv_idx += 1

        # 검출된 bbox가 없는 경우
        if len(csv_tmp_list) == 0:
            # 306번째 줄과 동일한데 함수로 어떻게 뺄 수 있을까?
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
                track_by_skeleton(json_obj, frame_num, skeleton_id)
                frame_num += 1
                is_frame_passed = True
            elif adult_id >= 0:
                saved_adult_id = adult_id
            continue

        image = cv2.imread(f"../output/video/{get_video_name()}/frames/{frame_num}.png")
        not_detected = True  # saved_adult_id 의 bbox가 감지되지 않았으면 True

        for csv_tmp_line in csv_tmp_list:
            # 추적 대상 tracking하며 관절 추출
            if int(csv_tmp_line[1]) == saved_adult_id:
                # cropped image 저장하고 skeleton extend하기
                save_cropped_image_bbox(csv_tmp_line[2:6], image, frame_num)
                extend_skeleton_list(get_skeleton_json(frame_num))
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
                track_by_skeleton(json_obj, frame_num, skeleton_id)
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
    # argument 에 따른 데이터 초기화
    original_video_name = sys.argv[1]
    set_video_name(original_video_name)
    init_point()
    init_rate()

    if sys.argv[2] == "n":
        original_video_path = f'../media/non-violence/{original_video_name}.mp4'
    else:
        original_video_path = f'../media/violence/{original_video_name}.mp4'

    # frame_size 정해줌
    cap = cv2.VideoCapture(original_video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    set_frame_size(int(width), int(height))
    cap.release()

    # MOT 으로 tracking 실행
    exec_MOT(original_video_path)

    # 성인 id 탐색
    BBOX_CSV_PATH = f'../output/video/{get_video_name()}/final/results{get_video_name()}_0.csv'
    adult_id = get_first_adult_id(BBOX_CSV_PATH)

    # 성인의 관절 데이터를 json 파일로 저장
    SKELETON_JSON_PATH = f'../output/video/{get_video_name()}/results{get_video_name()}.json'
    with open(SKELETON_JSON_PATH, 'w', encoding="utf-8") as make_file:
        json.dump(get_skeleton_list(), make_file, ensure_ascii=False, indent="\t")

    # 폭력 의심 구간을 영상으로 만듦
    make_violence_video(SKELETON_JSON_PATH)
