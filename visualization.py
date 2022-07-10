import numpy as np
import cv2 as cv
import copy

from constants import KEYPOINT_EDGE_INDS_TO_COLOR

def draw_frame(
        image,
        keypoints_scores,
        repCount,
        keypoint_score_th=0.11
    ):
    width, height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        point1 = (int(width * keypoints_scores[0][edge_pair[0]][0]), int(height * keypoints_scores[0][edge_pair[0]][1]))
        point2 = (int(width * keypoints_scores[0][edge_pair[1]][0]), int(height * keypoints_scores[0][edge_pair[1]][1]))
        score1 = keypoints_scores[1][edge_pair[0]]
        score2 = keypoints_scores[1][edge_pair[1]]
        if (score1 > keypoint_score_th and score2 > keypoint_score_th):
            cv.line(debug_image, point1, point2, color, 4)
            cv.circle(debug_image, point1, 6, color, -1)
            cv.circle(debug_image, point2, 6, color, -1)


    cv.putText(debug_image,
               f"Rep Count: {repCount}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2,
               cv.LINE_AA)
    return debug_image



