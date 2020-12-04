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


def region_selection(image_path:str):
    """
    변환 지역 설정
    """
    image = cv2.imread(image_path)
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
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_line(image_path:str):  
    image = cv2.imread(image_path)
    lines_down = [] # (x1, y1)
    lines_up = [] # (x2, y2)
    lines = [] # (x1,y1,x2,y2)
    rows, cols = image.shape[:2]
    for k in range(2) :
        for i in range(cols) : 
            if k==0 :
                if image[0,i] == (255,255,255) : 
                    lines_down.append(list(0,i))
            else :
                if image[5,i] == (255,255,255) : 
                    lines_up.append(list(0,i))
    for i in len(lines_down) : 
        lines.append(lines_down[i]+lines_up[i])
    return lines # x좌표와 y좌표로 정의된 선분들의 리스트 


def average_slope(lines):
 
    left_lines = []  # (slope)
    right_lines = []  # (slope)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if slope > 0: #기울기가 양수면 왼쪽에 위치한 선이다
                left_lines.append(slope)
            else:
                right_lines.append(slope)
    return left_lines,right_lines #(왼쪽 선의 기울기와 오른쪽 선의 기울기 도출 : (2,2),(-2,-2))


def get_line_gradient(left_lines,right_lines):
    left_gradient=(left_lines[0]+left_lines[1])/2
    right_gradient=(right_lines[0]+right_lines[1])/2
    return (left_gradient+right_gradient)/2


def get_direction(theta1: float, theta2: float):
    theta = (theta1+theta2)/2
    return theta


if __name__ == "__main__":
    read_video(f"./videos/20200925_091111.mp4", 200)
    show_canny(f"./images/frame_{100}.jpg")
