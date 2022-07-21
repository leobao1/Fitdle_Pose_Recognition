import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 as cv
import copy
import time

from math import atan2, degrees
from constants import KEYPOINT_DICT, EXERCISES
from visualization import draw_frame

# Import matplotlib libraries
# from matplotlib import pyplot as plt
# from matplotlib.collections import LineCollection
# import matplotlib.patches as patches

def _get_model(model_name):
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    return module.signatures['serving_default'], input_size


def _movenet_estimation(model, input_size, image):
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換
    input_image = input_image.reshape(-1, input_size, input_size, 3)  # リシェイプ
    input_image = tf.cast(input_image, dtype=tf.int32)  # int32へキャスト

    outputs = model(input_image)

    keypoints_with_scores = outputs['output_0'].numpy()
    keypoints_with_scores = np.squeeze(keypoints_with_scores)
    keypoints = []
    scores = []
    for index in range(17):
        keypoint_x = keypoints_with_scores[index][1]
        keypoint_y = keypoints_with_scores[index][0]
        score = keypoints_with_scores[index][2]

        keypoints.append([keypoint_x, keypoint_y])
        scores.append(score)

    return keypoints, scores


def _dist_between_points(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def _angle_between(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    return abs(deg2 - deg1) if abs(deg2 - deg1) <= 180 else abs(deg1 - deg2)


def _verify_output(keypoints_scores, expectedPose, threshold=0.11):
    diffs = {}
    for posture, expectedAngle in expectedPose.items():
        if 'both' in posture[0]:
            # Super hacky way to do this
            angle_r = 99999
            angle_l = 99999

            if (keypoints_scores[1][KEYPOINT_DICT.get(posture[0].replace('both', 'right'))] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[1].replace('both', 'right'))] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[2].replace('both', 'right'))] > threshold):
                p1_r = keypoints_scores[0][KEYPOINT_DICT.get(posture[0].replace('both', 'right'))]
                p2_r = keypoints_scores[0][KEYPOINT_DICT.get(posture[1].replace('both', 'right'))]
                p3_r = keypoints_scores[0][KEYPOINT_DICT.get(posture[2].replace('both', 'right'))]
                angle_r = _angle_between(p2_r, p1_r, p3_r)

            if (keypoints_scores[1][KEYPOINT_DICT.get(posture[0].replace('both', 'left'))] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[1].replace('both', 'left'))] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[2].replace('both', 'left'))] > threshold):
                p1_l = keypoints_scores[0][KEYPOINT_DICT.get(posture[0].replace('both', 'left'))]
                p2_l = keypoints_scores[0][KEYPOINT_DICT.get(posture[1].replace('both', 'left'))]
                p3_l = keypoints_scores[0][KEYPOINT_DICT.get(posture[2].replace('both', 'left'))]
                angle_l = _angle_between(p2_l, p1_l, p3_l)

            diffs[posture[0]] = min(abs(angle_r - expectedAngle), abs(angle_l - expectedAngle))
        else:
            if (keypoints_scores[1][KEYPOINT_DICT.get(posture[0])] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[1])] > threshold and 
                keypoints_scores[1][KEYPOINT_DICT.get(posture[2])] > threshold):
                p1 = keypoints_scores[0][KEYPOINT_DICT.get(posture[0])]
                p2 = keypoints_scores[0][KEYPOINT_DICT.get(posture[1])]
                p3 = keypoints_scores[0][KEYPOINT_DICT.get(posture[2])]
                angle = _angle_between(p2, p1, p3)
                error = abs(angle - expectedAngle)
                diffs[posture[0]] = error
    return diffs

def track(exercise, capture_device=0, height=540, width=960, model_name="movenet_thunder"):
    model, input_size = _get_model(model_name)

    cap = cv.VideoCapture(capture_device)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

    exerciseStates = EXERCISES.get(exercise)['states']
    allowed_err = EXERCISES.get(exercise)['allowed_err']
    alert_err = EXERCISES.get(exercise)['alert_err']
    state = 0
    numStates = len(exerciseStates)
    message = []
    curr_err = {}

    repCounts = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        keypoints_scores = _movenet_estimation(
            model,
            input_size,
            frame,
        )

        diffs_curr = _verify_output(keypoints_scores, exerciseStates[state])
        diffs_next = _verify_output(keypoints_scores, exerciseStates[(state+1)%numStates])

        best_pose = True
        for k, v in diffs_curr.items():
            if k not in curr_err:
                break
            elif curr_err[k] < v:
                best_pose = False
                break

        if best_pose: curr_err = diffs_curr

        if all(err < allowed_err for err in diffs_next.values()):
            message = []
            for k, v in curr_err.items():
                print(f"{k}: {v}")
                if v > alert_err:
                    message.append(f"Your form at your {k.replace('both_', '')} is a bit off")

            curr_err = {}
            state = (state + 1)%numStates
            if state == 0:
                repCounts += 1

        debug_image = draw_frame(
            frame,
            keypoints_scores,
            repCounts,
            message
        )

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        cv.imshow('MoveNet(singlepose) Demo', debug_image)
