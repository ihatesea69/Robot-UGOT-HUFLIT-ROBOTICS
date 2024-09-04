### [**Engineer Vehicle Catch Ball**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=engineer-vehicle-catch-ball)

### [**Overview**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=overview)

This is an example of a simple ball-grabbing demonstration using an Engineer Vehicle.

#### [**Preparation in advance**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=preparation-in-advance)

##### [**Hardware requirements:**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=hardware-requirements)

Engineer Vehicle, Requires camera connection and TOF Sensor

##### [**Software requirements:**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=software-requirements)

UGOT system version requirement 1.5.3.3 or above

#### [**Reference code**](https://docs.ubtrobot.com/ugot/#/en-us/extension/case/EngineerVehicleCatchBall?id=reference-code)

``` python
from ugot import ugot
import time

class Color:
    GREEN = "GREEN"
    BLUE = "BLUE"
    PURPLE = "PURPLE"
    RED = "RED"
    UN_SET = "UNSET"

camera_center_x = 320
camera_center_y = 240
max_rotate_speed = 60
max_forward_speed = 40

u = ugot.UGOT()
u.initialize("10.10.36.190")

# Get the color of the ball to be grabbed via dialogue
u.load_models(["color_recognition"])

u.show_light_rgb_effect(255, 100, 100, 0)
u.play_sound("bear")
u.play_sound("lion")


pid_rotate_speed = u.create_pid_controller()
pid_rotate_speed.set_pid(0.3, 0, 0.01)

pid_forward_speed = u.create_pid_controller()
pid_forward_speed.set_pid(0.9, 0, 0.001)


def main():
    try:
        while True:
            # target_color = listen_and_set_color()
            target_color = Color.GREEN
            go_and_grap_ball(target_color)

            # get_color_info()

    finally:
        u.mecanum_stop()

def get_color_info():
    color_info = u.get_color_total_info()
    print(color_info)

# Get the color of the ball to be grabbed via dialogue
def listen_and_set_color():
    target_color = Color.UN_SET

    u.mechanical_clamp_release()
    while target_color == Color.UN_SET:
        u.play_audio_tts("Please tell me whether to grab the blue ball or the green ball.", 0, True)
        heard_string = u.start_audio_asr()
        print(heard_string)

        if str(Color.GREEN) in heard_string:
            target_color = Color.GREEN
        elif str(Color.BLUE) in heard_string:
            target_color = Color.BLUE
        elif str(Color.RED) in heard_string:
            target_color = Color.RED
        elif str(Color.PURPLE) in heard_string:
            target_color = Color.PURPLE
        else:
            target_color = Color.UN_SET
            u.play_audio_tts("I didn't hear you about the colors.", 0, True)
        if target_color != Color.UN_SET:
            u.play_audio_tts(f"OK，I'll grab {str(target_color)} ball", 0, True)
    return target_color

# Begin the task of catching the ball
def go_and_grap_ball(target_color):
    start_y_speed = 0
    grab_color = target_color

    adjust_direction(start_y_speed, grab_color)
    reach_target(grab_color)
    grap_target()


# Calculate the speed of the cart turning in place
def get_rotate_speed(target_color):
    color_info = u.get_color_total_info()
    # print(color_info)
    [color, type, target_center_x, target_center_y, height, width, area] = color_info
    # print(str(target_color), color, str(target_color) == color)
    if (
        len(color) == 0
        or len(type) == 0
        or target_center_x == -1
        or str(target_color) != color
    ):
        target_center_x = 460

    gap = target_center_x - camera_center_x
    rotate_speed = round(pid_rotate_speed.update(gap))
    if rotate_speed > max_rotate_speed:
        rotate_speed = max_rotate_speed
    if rotate_speed < -max_rotate_speed:
        rotate_speed = -max_rotate_speed
    # print(rotate_speed)
    return rotate_speed


# Adjust trolley orientation
def adjust_direction(forward_speed, target_color):
    u.mechanical_arms_restory()
    u.mechanical_clamp_release()

    rotate_speed = get_rotate_speed(target_color)
    while abs(rotate_speed) > 2:
        u.mecanum_move_xyz(0, forward_speed, int(rotate_speed))
        rotate_speed = get_rotate_speed(target_color)


# Calculate forward speed (a negative number means backward speed)
def get_forward_speed(target_color):
    distance = u.read_distance_data(21)
    print("distance:", distance)
    forward_speed = round(pid_forward_speed.update(8 - distance))
    if forward_speed > max_forward_speed:
        forward_speed = max_forward_speed
    if forward_speed < -max_forward_speed:
        forward_speed = -max_forward_speed
    print("forward_speed:", forward_speed)
    return forward_speed


# Close to the target
def reach_target(target_color):
    forward_speed = 10
    while abs(forward_speed) > 1:
        # mecanum_translate_speed(angle, speed)
        direction = 0 if forward_speed > 0 else 1
        print("direction:", direction)
        u.mecanum_move_speed(direction, abs(forward_speed))
        forward_speed = get_forward_speed(target_color)
        adjust_direction(forward_speed, target_color)

    u.mecanum_stop()


# Grab the target
def grap_target():
    u.mechanical_clamp_release()
    u.mechanical_move_axis(13.8, -16.2, 0.005, 1500)
    time.sleep(2)
    u.mechanical_clamp_close()
    time.sleep(1)
    u.mechanical_move_axis(-3.9, 11.6, 0.005, 800)
    time.sleep(1)
    u.mecanum_turn_speed_times(2, 200, 2, 0)
    u.mechanical_move_axis(24.3, -3.2, 0, 150)
    u.play_audio_tts("Finish", 0, True)
    u.mechanical_arms_restory()
    u.mechanical_clamp_release()


if __name__ == "__main__":
    main()
```
