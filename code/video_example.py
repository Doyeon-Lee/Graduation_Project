import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np

body_point = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
              "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
              "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
              "REye", "LEye", "REar", "LEar", "LBigToe",
              "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel", "Background"]

body_dict = {}
for i, v in enumerate(body_point):
    body_dict[v] = i

# default frame size
FRAME_W = 0
FRAME_H = 0


# roi = region of interest
# bbox = bounding box?
def obj_tracking(_x, _y, _w, _h):
    # 트랙커 객체 생성자 함수 리스트 ---①
    trackers = [cv2.TrackerMIL_create,
                cv2.TrackerKCF_create,
                cv2.TrackerGOTURN_create,  # 버그로 오류 발생
                cv2.TrackerCSRT_create]
    trackerIdx = 0  # 트랙커 생성자 함수 선택 인덱스
    tracker = None
    isFirst = True

    # 비디오 파일 선택 ---②
    video_src = "../output/output.avi"
    cap = cv2.VideoCapture(video_src)

    FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
    delay = int(1000 / fps)
    # win_name = 'Tracking APIs'

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('../output/bbox_output2.avi', fourcc, 20.0, (FRAME_W, FRAME_H))

    while cap.isOpened():
        ret, frame = cap.read()
        # roi = (0, 0, 0, 0)

        if not ret:
            print('Cannot read video file')
            break
        img_draw = frame.copy()
        if tracker is None:  # 트랙커 생성 안된 경우
            cv2.putText(img_draw, "Press the Space to set ROI!!", \
                        (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            ok, bbox = tracker.update(frame)  # 새로운 프레임에서 추적 위치 찾기 ---③
            (_x, _y, _w, _h) = bbox
            if ok:  # 추적 성공
                cv2.rectangle(img_draw, (_x, _y), (_x + _w, _y + _h), (0, 255, 0), 2, 1)
            else:  # 추적 실패
                cv2.putText(img_draw, "Tracking fail.", (100, 80), \
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        trackerName = tracker.__class__.__name__
        cv2.putText(img_draw, str(trackerIdx) + ":" + trackerName, (100, 20), \
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

        out.write(img_draw)
        # cv2.imshow(win_name, img_draw)
        key = cv2.waitKey(delay) & 0xff

        # 비디오 파일 최초 실행 ---④
        if video_src != 0 and isFirst:
            isFirst = False
            roi = (_x, _y, _w, _h)  # 초기 객체 위치 설정
            if roi[2] and roi[3]:         # 위치 설정 값 있는 경우
                tracker = trackers[trackerIdx]()    # 트랙커 객체 생성 ---⑤
                isInit = tracker.init(frame, roi)

        # elif key in range(48, 52): # 0~7 숫자 입력   ---⑥
        #     trackerIdx = key-48     # 선택한 숫자로 트랙커 인덱스 수정
        #     if bbox is not None:
        elif key == 27:
            break
        else:
            # trackerIdx = key-48
            tracker = trackers[trackerIdx]()  # 선택한 숫자의 트랙커 객체 생성 ---⑦
            isInit = tracker.init(frame, roi)  # 이전 추적 위치로 추적 위치 초기화

    else:
        print("Could not open video")
    cap.release()
    cv2.destroyAllWindows()


# bbox 찾기1
def check_boundary(person_bodypoint, frame_w, frame_h):
    n = body_dict['Neck']
    ls = body_dict['LShoulder']
    rs = body_dict['RShoulder']
    mh = body_dict['MidHip']
    lh = body_dict['LHip']
    rh = body_dict['RHip']

    shoulder_len = max(person_bodypoint[n][0] - person_bodypoint[rs][0],
                       person_bodypoint[n][0] - person_bodypoint[ls][0])

    _x = max(0, int(person_bodypoint[n][0] - shoulder_len))
    _y = int(person_bodypoint[n][1])
    _w = 0
    _h = 0

    # MidHip이 안보일 경우
    if not person_bodypoint[mh][1]:
        _h = int(frame_h - person_bodypoint[n][1])
    else:
        _h = int(person_bodypoint[mh][1] - person_bodypoint[n][1])

    # Neck == 0
    if not person_bodypoint[n][0]:
        if not person_bodypoint[lh][0]:  # LHip == 0
            _w = min(person_bodypoint[rh][0], FRAME_W - person_bodypoint[rh][0])
        elif not person_bodypoint[rh][0]:
            _w = min(person_bodypoint[lh][0], FRAME_W - person_bodypoint[lh][0])
        else:
            _w = abs(person_bodypoint[rh][0] - person_bodypoint[lh][0])

    else:  # Neck != 0
        if not person_bodypoint[mh][0]:  # MidHip == 0
            if not person_bodypoint[ls][0]:  # LShoulder == 0
                _w = min(person_bodypoint[rs][0], FRAME_W - person_bodypoint[rs][0])
            elif not person_bodypoint[rs][0]:  # RShoulder == 0
                _w = min(person_bodypoint[ls][0], FRAME_W - person_bodypoint[ls][0])
        else:  # Neck != 0 and MidHip != 0
            _w = abs(person_bodypoint[ls][0] - person_bodypoint[rs][0])

    _w = int(_w)
    return _x, _y, _w, _h


# bbox 찾기1
def find_bbox(person_bodypoint):
    person_bodypoint = np.array(person_bodypoint)

    # 0을 제외하고 행의 최소값 구하기
    x_min, y_min, _ = np.apply_along_axis(lambda a: np.min(a[a != 0]), 0, person_bodypoint)
    x_max, y_max, _ = np.apply_along_axis(lambda a: np.max(a[a != 0]), 0, person_bodypoint)

    w = int(x_max - x_min)
    h = int(y_max - y_min)

    return int(x_min), int(y_min), int(w), int(h)


def get_location(frame_data):
    person_id = 0
    person_bodypoint = frame_data[0]['person'][person_id]['keypoint']

    tmp = []
    for key in body_point:
        tmp.append(person_bodypoint[key])

    dict_to_list = []
    for dict in tmp:
        tmp2 = []
        for k, v in dict.items():
            tmp2.append(v)
        dict_to_list.append(tmp2)

    _x, _y, _w, _h = find_bbox(dict_to_list) # check_boundary(dict_to_list, FRAME_W, FRAME_H)
    print(_x, _y, _w, _h)
    obj_tracking(_x, _y, _w, _h)


def make_json(datum, frame_id):
    keypoints_tmp = []
    keypoints = {'frame_id': frame_id}
    for person_id in range(datum.poseKeypoints.shape[0]):

        location = {}
        for keypoint in range(25):
            body = {
                "x": float(datum.poseKeypoints[person_id][keypoint][0]),
                "y": float(datum.poseKeypoints[person_id][keypoint][1]),
                "accuracy": float(datum.poseKeypoints[person_id][keypoint][2])
            }
            location.update({body_point[keypoint]: body})

        keypoints_tmp.append({
            "person_id": person_id,
            "keypoint": location
        })
    keypoints["person"] = keypoints_tmp
    return keypoints


def detect_skeleton():
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../openpose/build/python/openpose/Release');
                os.environ['PATH'] = os.environ[
                                         'PATH'] + ';' + '../../openpose/build/x64/Release;' + '../../openpose/build/bin'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../openpose/build/python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--video_path", default="../media/17.mp4", help="Read input video (avi, mp4).")
        parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../openpose/models/"
        params["disable_multi_thread"] = "false"
        numberGPUs = 1

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item

        # Construct it from system arguments
        # op.init_argv(args[1])
        # oppython = op.OpenposePython()

        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Video
        datum = op.Datum()
        cap = cv2.VideoCapture(args[0].video_path)

        FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        # out = cv2.VideoWriter('../output/bbox_output.avi', fourcc, 20.0, (FRAME_W, FRAME_H))

        frame_data = []
        frame_id = -1

        while cap.isOpened():
            frame_id += 1

            grabbed, frame = cap.read()

            if frame is None or not grabbed:
                print("Finish reading video frames...")
                break

            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            # # write the flipped frame
            # out.write(datum.cvOutputData)

            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            # cv2.waitKey(1)

            # save the keypoint as a list
            frame_data.append(make_json(datum, frame_id))

        else:
            print('cannot open the file')

        # # show list as json
        # with open('../output/output.json', 'w', encoding="utf-8") as make_file:
        #     json.dump(frame_data, make_file, ensure_ascii=False, indent="\t")

        cap.release()
        # out.release()
        # cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        sys.exit(-1)

    return frame_data
