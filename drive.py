from gpiozero import Motor

def control():  # :3
    motor = Motor(right=핀번호, left=핀번호)
    if get_line_gradient() < 90도 :
        motor.left(speed = 0.3) #todo
    elif get_line_gradient() > 90도 : 
        motor.right(speed = 0.3) #todo