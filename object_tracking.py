import numpy as np
import argparse
import cv2


frame = None
roi_points = []
input_mode = False


def select_roi(event, x, y, flags, param):
    global frame, roi_points, input_mode

    if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
        roi_points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", frame)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",
                    help="path to the (optional) video file")
    args = vars(ap.parse_args())

    global frame, roi_points, input_mode

    if not args.get("video", False):
        camera = cv2.VideoCapture(0)

    else:
        camera = cv2.VideoCapture(args["video"])

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", select_roi)

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    roi_box = None

    while True:

        grabbed, frame = camera.read()

        if not grabbed:
            break

        if roi_box is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            r, roi_box = cv2.CamShift(back_proj, roi_box, termination)
            pts = np.int0(cv2.boxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("i") and len(roi_points) < 4:

            input_mode = True
            orig_frame = frame.copy()

            while len(roi_points) < 4:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)

            roi_points = np.array(roi_points)
            s = roi_points.sum(axis=1)
            top_left = roi_points[np.argmin(s)]
            bottom_right = roi_points[np.argmax(s)]

            roi = orig_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            roi_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

        elif key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
