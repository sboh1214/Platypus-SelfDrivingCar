import time
import numpy as np
import cv2

def read_video(video_path: str, stop_frame: int) -> None:
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


def show_canny(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edge = cv2.Canny(img, 50, 200)
    cv2.imshow('Result Image', edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_line_gradient():
    switch=0
    tmp=0
    #canny에서 선분의 a, b, r, theta 받아오기
    for _ in range(0, 1) : #todo
        if a == XMAX:
            tmp=theta
            if switch==0:
                theta1=theta
                switch=1
    return (tmp+theta1)/2
        
    
def get_direction(theta1: float, theta2: float):
    theta = (theta1+theta2)/2

if __name__ == "__main__":
    read_video(f"./videos/20200925_091111.mp4", 1000)
    show_canny(f"./images/frame_{500}.jpg")
