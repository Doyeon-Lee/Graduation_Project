# Graduation_Project
## Violence Detection

1. PycharmProjects 밑에 프로젝트를 하나 생성
2. 1에서 만든 프로젝트 폴더 밑에 openpose를 설치하여 build까지 실행
3. 1에서 만든 프로젝트 폴더 밑에 FairMOT을 설치하여 build까지 실행(그 밑에 DCNv2도 설치해준다)
4. 이 리포지토리를 1에서 만든 프로젝트 폴더 밑에 다운로드 후 이름을 main으로 변경
___
___
### FairMOT 수정
___
#### track.py 수정
FairMOT/src/track.py의 25번째 줄에
```
import sys
sys.path.append('../../../../main/code')
from global_data import get_video_name
```
를 추가해 주고, 82번째 줄에
```
if not os.path.exists(f"../../main/output/video/{get_video_name()}/frames"):
    os.makedirs(f"../../main/output/video/{get_video_name()}/frames")
```
를, 85번째 줄에
```
cv2.imwrite(f"../../main/output/video/{get_video_name()}/frames/{frame_id}.png", img0)
```
를 추가해준다.
___
#### jde.py 수정
FairMOT/src/lib/datasets/dataset/jde.py의 98번째 줄에
```
self.w, self.h = 1920, 1080
```
를
```
self.w, self.h = img_size[0], img_size[1]
```
로 고친다. 원본 영상과 비슷한 사이즈를 유지하도록 하기 위해 수정하는 것이다.
___
#### opts.py 수정
FairMOT/src/lib/opts.py의 121번째 줄에
```
self.parser.add_argument('--input-video-name', type=str,
                             default='MOT16-03.mp4',
                             help='input video file name')
```
를 추가한다.