import os
import sys
import cv2
import json
import argparse
from sys import platform

from rw_json import *
from global_data import *

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
except Exception as e:
    print(e)
    sys.exit(-1)


def detect_skeleton(file_name, path='', input_type='video', frame_id=-1, cropped=False):
    # Flags
    parser = argparse.ArgumentParser()
    if input_type == 'photo':
        parser.add_argument("--image_path", default=f"../output/video/{file_name}/frame/00000.jpg",
                            help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    else:
        parser.add_argument("--video_path", default=f"../media/{file_name}.mp4", help="Read input video (avi, mp4).")

    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args(path)

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../openpose/models/"
    params["model_pose"] = "MPI"
    params["disable_multi_thread"] = "false"
    params["net_resolution"] = "-1x160"

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

    datum = op.Datum()
    if input_type == 'photo':
        imageToProcess = cv2.imread(args[0].image_path)
        if not cropped:
            imageToProcess = cv2.resize(imageToProcess, get_frame_size(), interpolation=cv2.INTER_CUBIC)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # save the keypoint as a list
        picture_json = make_skeleton_json(datum, frame_id)

        # 사람이 한 명도 잡히지 않는 frame 예외처리
        if picture_json is None:
            one_frame_data = {'frame_id': frame_id}
            location = {}
            for keypoint in range(15):
                body = {
                    "x": float(0),
                    "y": float(0),
                    "accuracy": float(0)
                }
                location.update({body_point[keypoint]: body})

            one_frame_data["person"] = [{
                "person_id": 0,
                "keypoint": location
            }]
            picture_json = [one_frame_data]
        else:
            picture_json = [picture_json]

        with open(f'../output/video/{file_name}/output{file_name}_0.json', 'w', encoding="utf-8") as make_file:
            json.dump(picture_json, make_file, ensure_ascii=False, indent="\t")

        # return frame_data
        return f'../output/video/{file_name}/output{file_name}_0.json'
    else:
        # Process Video
        cap = cv2.VideoCapture(args[0].video_path)
        set_frame_size(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 관절을 입힌 동영상 생성
        # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        # out = cv2.VideoWriter(f'../output/output/video/{file_name}.avi', fourcc, 20.0, get_frame_size())

        frame_data = []

        while cap.isOpened():
            frame_id += 1

            grabbed, frame = cap.read()

            if frame is None or not grabbed:
                print("Finish reading video frames...")
                break

            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))
            # print("Body keypoints: \n" + str(datum.poseKeypoints))

            # 동영상 저장
            #out.write(datum.cvOutputData)

            # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            # cv2.waitKey(1)

            # save the keypoint as a list
            one_frame_data = make_skeleton_json(datum, frame_id)

            # 사람이 한 명도 잡히지 않는 frame 예외처리
            if one_frame_data is None:
                one_frame_data = {'frame_id': frame_id}
                location = {}
                for keypoint in range(15):
                    body = {
                        "x": float(0),
                        "y": float(0),
                        "accuracy": float(0)
                    }
                    location.update({body_point[keypoint]: body})

                one_frame_data["person"] = [{
                    "person_id": 0,
                    "keypoint": location
                }]

            frame_data.append(one_frame_data)

        else:
            print('cannot open the file')

        # show list as json
        # with open(f'../output/json/output{file_name}.json', 'w', encoding="utf-8") as make_file:
        with open(f'../output/video/{file_name}/output{file_name}_0.json', 'w', encoding="utf-8") as make_file:
            json.dump(frame_data, make_file, ensure_ascii=False, indent="\t")

        cap.release()
        # out.release()
        # cv2.destroyAllWindows()

        # return frame_data
        return f'../output/video/{file_name}/output{file_name}_0.json'


# detect_skeleton("54")

