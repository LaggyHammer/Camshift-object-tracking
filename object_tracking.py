# coding: utf-8
# =====================================================================
#  Filename:    object_tracking.py
#
#  py Ver:      python 3.6 or later
#
#  Description: Recognizes regions of text in a given image
#
#  Usage: python object_detection.py
#         or
#         python object_detection.py --video test.mov
#
#  Author: Ankit Saxena (ankch24@gmail.com)
# =====================================================================

import numpy as np
import argparse
import cv2

# global variables to be used
frame = None
roi_points = []
input_mode = False


def get_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help='path to optional video file')
    args = vars(ap.parse_args())

    return args


def select_roi(event, x, y, flags, param):
    """
    Draw circles at the selection region vertices and show the selected ROI on image
    :param event: mouse callback event from openCV
    :param x: x coordinate of pointer
    :param y: y coordinate of pointer
    """
    global frame, roi_points, input_mode

    # append to points only if left mouse button is clicked at the position
    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def frame_roi():
    """
    Freezes frame on entering insert mode. Upon selecting ROI, converts ROI to HSV color space
    and calculates its hue histogram
    :return: roi histogram, roi box (tuple)
    """

    global frame, roi_points, input_mode

    input_mode = True
    orig_frame = frame.copy()

    # only frame when 4 points are not selected
    while len(roi_points) < 4:
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    roi_points = np.array(roi_points)
    s = roi_points.sum(axis=1)
    top_left = roi_points[np.argmin(s)]
    bottom_right = roi_points[np.argmax(s)]

    # convert ROI to HSV color space
    roi = orig_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # calculate histogram for the ROI & normalize it
    roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    roi_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

    return roi_hist, roi_box


def apply_camshift(roi_box, termination, roi_hist):
    """
    Applies the camshift algorithm to a back projection of the HSV color space ROI
    :param roi_box: region of interest box
    :param termination: termination criteria for the camshift algorithm iterations
    :param roi_hist: region of interest histogram
    """
    global frame, roi_points, input_mode

    # calculate back projection for the ROI
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply the camshift algorithm to the ROI
    r, roi_box = cv2.CamShift(back_projection, roi_box, termination)
    points = np.int0(cv2.boxPoints(r))
    cv2.polylines(frame, [points], True, (0, 255, 0), 2)


def main():

    args = get_arguments()

    global frame, roi_points, input_mode

    if not args.get("video", False):
        # start web cam feed
        camera = cv2.VideoCapture(0)

    else:
        # load video file
        camera = cv2.VideoCapture(args["video"])

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", select_roi)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roi_box = None

    # main loop
    while True:

        grabbed, frame = camera.read()

        if not grabbed:
            break

        # apply camshift if ROI is selected
        if roi_box is not None:
            apply_camshift(roi_box, termination, roi_hist)

        # show the feed/results
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # insert mode to select ROI
        if key == ord("i") and len(roi_points) < 4:
            roi_hist, roi_box = frame_roi()

        # quit if 'q' is pressed
        elif key == ord("q"):
            break

    # clean up endpoints
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
