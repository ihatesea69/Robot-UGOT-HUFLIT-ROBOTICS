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
```python
import cv2
import time
import numpy as np
import math
from pyzbar.pyzbar import decode
from ugot import ugot

got = ugot.UGOT()

got.initialize('172.31.79.160')
got.load_models(['line_recognition'])
got.set_track_recognition_line(0)

Pid_error=ugot.PID()
Pid_error.set_pid(0.15,0,0)


class Image:

    def __init__(self):
        self.prev_cX = None
        self.offset = None
        self.dir = None
        self.middleY = None
        self.middleX = None
        self.contours = None
        self.width = None
        self.height = None
        self.prev_MC = None
        self.image = None
        self.original_image = None
        self.contourCenterX = 0
        self.MainContour = None
        self.distance = []

    def Process(self):
        #imgray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to Gray Scale
        ret, thresh = cv2.threshold(self.image, 100, 200, cv2.THRESH_BINARY_INV)  # Get Threshold

        self.contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Get contour
   


        self.prev_MC = self.MainContour
        if self.contours:
            self.MainContour = max(self.contours, key=cv2.contourArea)
            
            
            leftmost = tuple(self.MainContour[self.MainContour[:, :, 0].argmin()][0])
            rightmost = tuple(self.MainContour[self.MainContour[:, :, 0].argmax()][0])
                    
            self.distance = np.linalg.norm(np.array(leftmost) - np.array(rightmost))
            

            self.height, self.width = self.image.shape[:2]

            self.middleX = int(self.width / 2)  # Get X coordinate of the middle point
            self.middleY = int(self.height / 2)  # Get Y coordinate of the middle point

            self.prev_cX = self.contourCenterX
            if self.getContourCenter(self.MainContour) != 0:
                self.contourCenterX = self.getContourCenter(self.MainContour)[0]
                if abs(self.prev_cX - self.contourCenterX) > 5:
                    self.correctMainContour(self.prev_cX)
            else:
                self.contourCenterX = 0

            #self.dir = int((self.middleX - self.contourCenterX) * self.getContourExtent(self.MainContour))
            weight = self.getContourExtent(self.MainContour)
            if weight < 0.5:
                self.offset= None
                return
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.offset = self.middleX - self.contourCenterX
            # Marked on image
            cv2.drawContours(self.image, self.MainContour, -1, (0, 255, 0), 3)  # Draw Contour GREEN
            cv2.circle(self.image, (self.contourCenterX, self.middleY), 7, (255, 255, 255),
                       -1)  # Draw dX circle WHITE
            cv2.circle(self.image, (self.middleX, self.middleY), 3, (0, 0, 255), -1)  # Draw middle circle RED

            cv2.putText(self.image, str(self.middleX - self.contourCenterX),
                        (self.contourCenterX + 20, self.middleY),
                        font, 1, (200, 0, 200), 2, cv2.LINE_AA)
            cv2.putText(self.image, "Weight:%.3f" % weight,
                        (self.contourCenterX + 20, self.middleY + 35), font, 0.5, (200, 0, 200), 1, cv2.LINE_AA)
            # Marked on original image
            cv2.drawContours(self.original_image, self.MainContour, -1, (0, 255, 0), 3)  # Draw Contour GREEN
            cv2.circle(self.original_image, (self.contourCenterX, self.middleY), 7, (255, 255, 255), -1)  # Draw dX circle WHITE
            cv2.circle(self.original_image, (self.middleX, self.middleY), 3, (0, 0, 255), -1)  # Draw middle circle RED

            cv2.putText(self.original_image, str(self.middleX - self.contourCenterX), (self.contourCenterX + 20, self.middleY),
                        font, 1, (200, 0, 200), 2, cv2.LINE_AA)
            cv2.putText(self.original_image, "Weight:%.3f" % weight,
                        (self.contourCenterX + 30, self.middleY + 15), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            
            #cv2.imshow("frame", self.image)
            #cv2.imshow("image", self.original_image)
            #cv2.waitKey(0)

    def getContourCenter(self, contour):
        M = cv2.moments(contour)

        if M["m00"] == 0:
            return 0

        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])

        return [x, y]

    def getContourExtent(self, contour):
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h
        if rect_area > 0:
            return float(area) / rect_area

    def Aprox(self, a, b, error):
        if abs(a - b) < error:
            return True
        else:
            return False

    def correctMainContour(self, prev_cx):
        if abs(prev_cx - self.contourCenterX) > 5:
            for i in range(len(self.contours)):
                if self.getContourCenter(self.contours[i]) != 0:
                    tmp_cx = self.getContourCenter(self.contours[i])[0]
                    if self.Aprox(tmp_cx, prev_cx, 5) is True:
                        self.MainContour = self.contours[i]
                        if self.getContourCenter(self.MainContour) != 0:
                            self.contourCenterX = self.getContourCenter(self.MainContour)[0]


def SlicePart(original_frame, processed_frame, images, n_slices, slices):
    offset = []
    distance = []
    height, width = processed_frame.shape[:2]
    sl = int(height / n_slices)
    for i in range(n_slices):
        part = sl * i
        crop_img = processed_frame[part:part + sl, 0:width]
        images[i].image = crop_img
        images[i].original_image = original_frame[part:part + sl, 0:width]
        if i >= (n_slices-slices):
            images[i].Process()
            offset.append(images[i].offset)
            distance.append(images[i].distance)
    return offset, distance


def RepackImages_Color(images):
    img = images[0].original_image
    for i in range(len(images)):
        if i == 0:
            img = np.concatenate((img, images[1].original_image), axis=0)
        if i > 1:
            img = np.concatenate((img, images[i].original_image), axis=0)

    return img

def RepackImages_Cropped(images):
    img = images[0].image
    for i in range(len(images)):
        if i == 0:
            img = np.concatenate((img, images[1].image), axis=0)
        if i > 1:
            img = np.concatenate((img, images[i].image), axis=0)

    return img


def Center(moments):
    if moments["m00"] == 0:
        return 0

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])

    return x, y


def remove_background(image, lower=0, upper=100):
    mask = cv2.inRange(image, lower, upper)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.bitwise_not(image, image, mask=mask)
    image = (255 - image)
    return image


def region_selection(image, top_row=0):
    """
    Determine and cut the region of interest in the input image.
    Parameters:
        image: we pass here the output from canny where we have
        identified edges in the frame
        top_row: top row index of the selected region
    """
    # create an array of the same size as of the input image
    mask = np.zeros_like(image)
    ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    # we have created this polygon in accordance to how the camera was placed
    height, width = image.shape[:2]
    top_left = [width * 0.2, top_row]
    top_right = [width * 0.8, top_row]
    bottom_left = [width * 0.05, height]
    bottom_right = [width * 0.95, height]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only region the of interest in the input image
    masked_image = cv2.bitwise_and(image, mask)
    mask = cv2.inRange(masked_image, 0, 0)
    masked_image = cv2.bitwise_or(masked_image, mask)
    return masked_image


def get_frame_offset(frame, n_slices=10, slices=7):
    gray_frame = frame
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Đổi hình sang xám
    height, width = gray_frame.shape[:2]
    top_row = height-(height/n_slices*slices)
    # A mask polygon to only focus on the road
    masked_frame= region_selection(gray_frame, top_row)
    removebg_frame = remove_background(masked_frame, 0, 200)

    Images = []
    for q in range(n_slices):
        Images.append(Image())
    if frame is not None:
        offset, distance = SlicePart(frame, removebg_frame, Images, n_slices, slices)
        fm = RepackImages_Color(Images) #RepackImages_Cropped(): cắt hình
        # cv2.imshow("Vision Race", fm)
        # cv2.waitKey(0)
        return offset, fm, distance

def TurnwSchedule(schedule2):
  if schedule2[0] == 2:
    got.transform_move_turn(0, 20, 2, 70)
    time.sleep(1)
    #global schedule 
    schedule= schedule[1:]
    print("-------------------------------", schedule)
  elif schedule2[0] == 3:
    got.transform_move_turn(0, 20, 3, 70)
    time.sleep(1)
    #global schedule 
    schedule= schedule[1:]
    print("-------------------------------", schedule)
  else:
    got.transform_move_speed(0,20)
    time.sleep(1)
    #global schedule 
    schedule= schedule[1:]
    print("-------------------------------", schedule)

got.open_camera()
schedule = [2,None,2,2]
speed = 20
while True:
  frame = got.read_camera_data()
  if frame is not None:
    nparr = np.frombuffer(frame, np.uint8)
    data = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    image = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    decoded_text = decode(image)
    for obj in decoded_text:
        qr_text = obj.data.decode("utf-8")
        print("Data:", qr_text)
    offset, fm, distance = get_frame_offset(data,10,7)
    cv2.imshow("Ugot",fm)
    print(offset)
    count = 0
    sum1 = 0
    for i in range(0,7):
      if offset[i] != None:
        sum1 = sum1 + offset[i]
        count = count + 1
    avr = sum1/count
    print(avr)
    if distance[6]:
      print(round(distance[6]))
    if offset[0] == None:
      speed = 5
    else:
      speed = 15
    dic=round(Pid_error.update(avr))
    if distance[6] > 350 and avr < 10:
      TurnwSchedule(schedule)
    if dic>=0:
      got.transform_move_turn(0, speed, 3, dic)
    else:
      got.transform_move_turn(0, speed, 2, abs(dic))
    cv2.waitKey(1)
    
```
