from cmath import sqrt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2 as cv
import copy
import time
import constants

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
    return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# p1 is vertex
# uses cosine law
def _angle_between_points(p1, p2, p3):
    d12 = _dist_between_points(p1, p2)
    d23 = _dist_between_points(p2, p3)
    d13 = _dist_between_points(p1, p3)
    return np.arccos((d12**2 + d13**2 - d23**2)/(2*d12*d13))


def _verify_output(keypoints_scores, expected, allowed_error=0.05):
    for pose in expected:
        p1 = keypoints_scores[pose[0]]
        p2 = keypoints_scores[pose[1]]
        p3 = keypoints_scores[pose[2]]
        angle = _angle_between_points(p1, p2, p3)
        error = abs(angle - pose[3])/pose[3]
        if (error > allowed_error):
            return False

    return True

def track(exercise, capture_device=0, height=540, width=960, model_name="movenet_lightning"):
    model, input_size = _get_model(model_name)

    cap = cv.VideoCapture(capture_device)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

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

        # _verify_output(keypoints_scores)

        print(keypoints_scores)
