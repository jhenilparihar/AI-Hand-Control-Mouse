import cv2
import numpy as np
import HandTracking as ht
import pyautogui

##############################################
cam_width, cam_height = 640, 480             #
previous_time = 0                            #
frame_red = 150     # Frame Reduction        #
smoothening = 2                              #
##############################################

pre_loc_x, pre_loc_y = 0, 0
cur_loc_x, cur_loc_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = ht.HandDetector(max_hands=1, detection_confidence=0.70)

screen_width, screen_height = pyautogui.size()
# print(screen_width, screen_height)

pyautogui.FAILSAFE = False

click_flag = False

while True:
    # Find hand landmark
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    landmark_list, bounding_box = detector.find_position(img)

    # Get the tip of the index and middle fingers

    if len(landmark_list) != 0:
        x1, y1 = landmark_list[0][1:]
        y1 = y1 - 50

        # Check which fingers are up

        fingers = detector.fingers_up()

        cv2.rectangle(img, (frame_red, frame_red+100), (cam_width - frame_red, (cam_height - frame_red)+100),
                      (255, 0, 0), 2)

        # Moving mode

        if fingers[1] == 1 or fingers[0] == 1:

            # Convert Coordinates

            x3 = np.interp(x1, (frame_red, cam_width-frame_red), (0, screen_width))
            y3 = np.interp(y1, (frame_red+100, (cam_height-frame_red)+100), (0, screen_height))

            # Smoothing Values
            cur_loc_x = pre_loc_x + (x3 - pre_loc_x)/smoothening
            cur_loc_y = pre_loc_y + (y3 - pre_loc_y)/smoothening

            # Move Mouse
            pyautogui.moveTo(cur_loc_x, cur_loc_y)
            cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
            pre_loc_x = cur_loc_x
            pre_loc_y = cur_loc_y

            # Left Clicking Mode
            if fingers[1] == 1 and fingers[2] == 1 and click_flag is True:

                # Find distance between fingers
                length, img, line_info = detector.find_distance(8, 12, img, t=2, r=7)

                if length < 35:
                    cv2.circle(img, (line_info[-2], line_info[-1]), 7, (0, 255, 0), cv2.FILLED)

                    # Click the mouse if distance short
                    pyautogui.click()
                    click_flag = False

            # Double Click
            if fingers[0] == 0 and fingers[1] == 1 and click_flag is True:
                pyautogui.doubleClick()
                click_flag = False

            # Right Click
            if fingers[0] == 1 and fingers[1] == 0 and click_flag is True:
                pyautogui.rightClick()
                click_flag = False

            if fingers[1] == 1 and fingers[0] == 1:
                click_flag = True

    # Frame Rate
    previous_time = detector.frame_rate(img, previous_time)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
