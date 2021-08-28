# Graduation_Project
Violence Detection

1. PycharmProjects 밑에 프로젝트를 하나 생성
2. 1에서 만든 프로젝트 폴더 밑에 openpose를 설치하여 build까지 실행
3. 1에서 만든 프로젝트 폴더 밑에 FairMOT을 설치하여 build까지 실행(그 밑에 DCNv2도 설치해준다)
4. 이 리포지토리를 1에서 만든 프로젝트 폴더 밑에 다운로드 후 이름을 main으로 변경
5. FairMOT/src/track.py의 25번째 줄에 import sys
sys.path.append('../../../../main/code')
from global_data import get_video_name를 추가해 주고, 83번째 줄에 cv2.imwrite(f"../../main/output/video/{get_video_name()}/frames/{frame_id}.png", img0)를 추가해준다.
