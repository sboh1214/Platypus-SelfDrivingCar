from gpiozero import Motor

########################################
#get_line_gradient=get_direction 입니다#
########################################
def control():  # :3
    motor = Motor(right=핀번호, left=핀번호)
    if get_direction() < 90도 :
        motor.left(speed = 0.3) #todo
    elif get_direction() > 90도 : 
        motor.right(speed = 0.3) #todo