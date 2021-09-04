import cv2


def get_skeleton(line, image):
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
    cv2.imwrite(f"../output/video/800/cropped_image/1944.png", cropped_image)


image = cv2.imread(f"../output/video/800/frames/1944.png")
get_skeleton([85.336677, 3.320668984, 273.6845987, 706.183255], image)
