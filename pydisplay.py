import cv2
import mediapipe as mp
import numpy as np
import keyboard
import subprocess
import sys

def spawn_program_and_die(program, exit_code=0):

    subprocess.Popen(program)
    sys.exit(exit_code)



mode = "Push-ups"
WINDOW_NAME = 'Mediapipe Feed'
pushup_stage = 'up'
situp_stage = 'up'
pushup_counter = 0
situp_counter = 0

display = pushup_counter

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()


        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates - SITUPS

            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]


            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Get coordinates - PUSHUPS
            l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle
            l_angle = int(calculate_angle(l_shoulder, l_elbow, l_wrist))
            r_angle = int(calculate_angle(r_shoulder, r_elbow, r_wrist))
            rfoot_angle = int(calculate_angle(r_hip, r_knee, r_ankle))
            lfoot_angle = int(calculate_angle(l_hip, l_knee, l_ankle))

            # Visualize angle
#            cv2.putText(image, str(rfoot_angle),
 #                       tuple(np.multiply(r_knee    , [640, 480]).astype(int)),
  #                      cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
   #                     )
    #        cv2.putText(image, str(lfoot_angle),
     #                   tuple(np.multiply(l_knee, [640, 480]).astype(int)),
      #                  cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
       #                 )
        #    cv2.putText(image, str(l_angle),
         #               tuple(np.multiply(l_elbow, [640, 480]).astype(int)),
          #              cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
           #             )
           # cv2.putText(image, str(r_angle),
            #            tuple(np.multiply(r_elbow, [640, 480]).astype(int)),
             #           cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
              #          )


            if r_angle > 100 and l_angle > 100:
                pushup_stage = 'up'
            if r_angle <= 100 and l_angle <= 100 and pushup_stage == 'up':
                pushup_stage = 'down'
                pushup_counter += 1
                print(pushup_counter)


            if rfoot_angle > 95 and lfoot_angle > 95:
                situp_stage = 'up'
            if rfoot_angle <= 95 and lfoot_angle <= 95 and situp_stage == 'up':
                situp_stage = 'down'

                situp_counter += 1
                print(situp_counter)

        except:
            pass

        cv2.rectangle(image, (0,0), (240, 73), (0,0,0), -1)

        cv2.putText(image, 'Press (r) to reset', (430, 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, 'Press (p) for push-ups', (430, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.putText(image, 'Press (s) to sit-ups', (430, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(image, (mode),
                    (10, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(display),
                    (190, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if keyboard.is_pressed('q'):
            break

        if keyboard.is_pressed('r'):
            pushup_counter = 0
            situp_counter = 0


        if keyboard.is_pressed('p'):
            if mode == "Sit-ups":
                mode = 'Push-ups'
                pushup_counter = 0
                situp_counter = 0
        if cv2.waitKey(1) == ord('s') and mode == "Push-ups":
            display = situp_counter
            mode = 'Sit-ups'
            pushup_counter = 0
            situp_counter = 0

        if mode == "Push-ups":
            display = pushup_counter
        if mode == "Sit-ups":
            display = situp_counter
    cap.release()
    cv2.destroyAllWindows()


