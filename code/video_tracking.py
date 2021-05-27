from global_data import *


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

    set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)  # 프레임 수 구하기
    delay = int(1000 / fps)
    win_name = 'Tracking APIs'

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('../output/bbox_output2.avi', fourcc, 20.0, get_frame_size())

    while cap.isOpened():
        ret, frame = cap.read()

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
        cv2.imshow(win_name, img_draw)
        key = cv2.waitKey(delay) & 0xff

        # 비디오 파일 최초 실행 ---④
        if video_src != 0 and isFirst:
            isFirst = False
            roi = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정
            # roi = (_x, _y, _w, _h)  # 초기 객체 위치 설정
            if roi[2] and roi[3]:         # 위치 설정 값 있는 경우
                tracker = trackers[trackerIdx]()    # 트랙커 객체 생성 ---⑤
                isInit = tracker.init(frame, roi)

        elif key in range(48, 52): # 0~7 숫자 입력   ---⑥
            trackerIdx = key-48     # 선택한 숫자로 트랙커 인덱스 수정
            if bbox is not None:
                tracker = trackers[trackerIdx]()  # 선택한 숫자의 트랙커 객체 생성 ---⑦
                isInit = tracker.init(frame, bbox)  # 이전 추적 위치로 추적 위치 초기화
        elif key == 27:
            break
        # else:
        #     # trackerIdx = key-48
        #     tracker = trackers[trackerIdx]()  # 선택한 숫자의 트랙커 객체 생성 ---⑦
        #     isInit = tracker.init(frame, roi)  # 이전 추적 위치로 추적 위치 초기화

    else:
        print("Could not open video")
    cap.release()
    cv2.destroyAllWindows()


# bbox 찾기1
def find_bbox(person_bodypoint, frame_w, frame_h):
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
def find_bbox2(person_bodypoint):
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

    _x, _y, _w, _h = find_bbox(dict_to_list)  # check_boundary(dict_to_list, FRAME_W, FRAME_H)
    print(_x, _y, _w, _h)
    obj_tracking(_x, _y, _w, _h)


def yolo(frame):
    # YOLO 가중치 파일과 CFG 파일 로드
    YOLO_net = cv2.dnn.readNet("../yolo/yolov2-tiny.weights", "../yolo/yolov2-tiny.cfg")
    # YOLO NETWORK 재구성
    classes = []
    with open("../yolo/yolo.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in YOLO_net.getUnconnectedOutLayers()]

    h, w, c = frame.shape

    # YOLO 입력
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 검출 신뢰도
            if confidence > 0.5:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화되어있으므로 다시 전처리
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 투영
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
    return frame
