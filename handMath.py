import time
import cv2
import hand_tracking as htm

# Set up camera dimensions
wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize variables
pTime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
first_number = None
second_number = None
hand_moved = False

# Main loop
while True:
    # Read frame from camera
    success, img = cap.read()

    # Detect hands in the frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    # Check hand movements
    if first_number is not None and len(lmList) == 0:
        hand_moved = True

    if len(lmList) != 0:
        fingers = []

        # Check each finger for openness
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)

        # Store finger count as numbers
        if first_number is None:
            first_number = totalFingers
            print("First number picked up:", first_number)
        elif second_number is None and hand_moved:
            second_number = totalFingers
            print("Second number picked up:", second_number)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display the image
    cv2.imshow("Image", img)
    cv2.waitKey(1)

    # Break out of the loop if both numbers are found
    if first_number is not None and second_number is not None:
        print("The sum of the numbers is:", first_number + second_number)
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
