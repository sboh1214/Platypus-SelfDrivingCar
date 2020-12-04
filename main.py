import time
import numpy as np
import cv2

XMAX = 480

def read_video(video_path: str, stop_frame: int) -> None:
    """
    영상을 읽어서 사진으로 변환
    """
    vidcap = cv2.VideoCapture(video_path)
    count = 0

    while(vidcap.isOpened()):
        # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
        # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
        # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
        _, image = vidcap.read()

        # 캡쳐된 이미지를 저장하는 함수
        cv2.imwrite(f"./images/frame_{count}.jpg", image)

        print(f'Saved frame_{count}.jpg')
        count += 1

        if count > stop_frame:
            break

    vidcap.release()

    #ㅇㅅㅇ#


def show_canny(image_path: str):
    pre_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = region_selection(pre_img)  # 변환 지역 설정 후 처리
    edge = cv2.Canny(img, 50, 200)  # 변환된 edge의 색 : (255,255,255)
    cv2.imshow('Result Image', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def region_selection(image):
    """
    변환 지역 설정
    """
    mask = np.zeros_like(image)

    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_line():  # todo
    pass


def average_slope_intercept(lines):
    """
    교점과 각도를 찾는다
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    left_lane = np.dot(left_weights,  left_lines) / \
        np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / \
        np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def get_line_gradient():
    switch = 0
    theta2 = 0
    get_line_info()
    for _ in range(0, 1):  # todo
        if a == XMAX:
            theta2 = theta
            if switch == 0:
                theta1 = theta
                switch = 1
    return (theta1+theta2)/2


def get_direction(theta1: float, theta2: float):
    theta = (theta1+theta2)/2
    return theta


if __name__ == "__main__":
    read_video(f"./videos/20200925_091111.mp4", 200)
    show_canny(f"./images/frame_{100}.jpg")
