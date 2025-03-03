# Steven8686, Redem-cat YOLOv5-based vehicle orientation detect, AGPL-3.0 license
"""
When circles (tires) are projected, it becomes ellipses. This part gives 4 methods to detect ellipses.
"""
import cv2
from skimage import data, draw, color, transform, feature
from scipy.optimize import leastsq
from assist_function import paint_contour, interpolate_contour_gap
import numpy as np
import math


def lsm(x, y):
    """
    Least square method of ellipse.
    Args:
        x (list(float/int)): x coords of data
        y (list(float/int)): y coords of data
    Return:
        fitted_params (tuple): lsm-fitted ellipse parameters. Elements are defined in ellipse_residuals.
    """
    def ellipse_residuals(params, x, y):
        a, b, cx, cy, phi = params
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        return ((x - cx) * cos_phi + (y - cy) * sin_phi) ** 2 / a ** 2 + \
            ((x - cx) * sin_phi - (y - cy) * cos_phi) ** 2 / b ** 2 - 1

    # Initial guess for the parameters [a, b, cx, cy, phi]
    initial_guess = [np.max(x) - np.min(x), np.max(y) - np.min(y),
                     np.mean(x), np.mean(y), 0]

    fitted_params, _ = leastsq(ellipse_residuals, initial_guess, args=(x, y))
    return fitted_params


def detect_ellipses_lsm(image):
    """
    Recognized as "lsm" in detect.py
    Generate binary image and obtain its edge using canny, then do lsm for all pixels.
    Args:
        image (ndarray): image of tire.
    Return:
        fitted_params (tuple): lsm-fitted ellipse parameters.
    """
    image_gray = color.rgb2gray(image)
    edges = feature.canny(image_gray, sigma=0.3, low_threshold=0.1, high_threshold=1.0)
    rows, cols = np.nonzero(edges)
    x = cols.tolist()
    y = rows.tolist()
    if len(x) == 0:
        return None, None, None
    a, b, cx, cy, phi = lsm(x, y)
    a = abs(a)
    b = abs(b)
    axis = (int(a), int(b))  # Note that opencv use (w, h)
    center = (int(cx), int(cy))
    orientation = math.degrees(phi) % 360

    return center, axis, orientation


def detect_ellipses_hough(image):
    """
    Recognized as "hough" in detect.py
    Generate binary image and obtain its edge using canny, then do hough transformation for all pixels.
    Hough transform is very time-consuming and unlikely to use in live.
    Args:
        image (ndarray): image of tire.
    Return:
        fitted_params (tuple): hough transform ellipse parameters.
    """
    h, w = image.shape[:2]
    image_gray = color.rgb2gray(image)
    edges = feature.canny(image_gray, sigma=0.5, low_threshold=0.1, high_threshold=1.0)
    result = transform.hough_ellipse(edges, accuracy=20, threshold=50, min_size=int(w*0.7), max_size=h)
    result.sort(order='accumulator')

    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]
    center = (xc, yc)
    axis = (a, b)
    # 在原图上画出椭圆
    # cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
    # image[cy, cx] = (0, 0, 255)  # 在原图中用蓝色表示检测出的椭圆
    # cv2.imwrite("checked_ellip1.jpg", image)
    return center, axis, orientation


def detect_ellipses_enhanced_hough(image):
    """
    Recognized as "e-hough" in detect.py
    Find contours and encircle them, then use the one with max area
    Caution: Due to unknown reasons, hough transform below may sometimes jam for very long
    Args:
        image (ndarray): image of tire.
    Return:
        fitted_params (tuple): hough transform ellipse parameters.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(magnitude_normalized, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    likely_contour = []
    for contour in contours:
        contour = interpolate_contour_gap(contour, max_gap=2)
        if cv2.contourArea(contour) < w*h*0.1 or cv2.contourArea(contour) > w*h*0.9:  # Tire should occupy a large area in cropped image. Ignore those too small.
            continue
        if len(contour) <= 8:
            continue

        likely_contour.append(contour)
    if len(likely_contour) > 0:
        sorted_contours = sorted(likely_contour, key=cv2.contourArea, reverse=True)
        ellip = sorted_contours[0]

        # convert contours to matrix for hough transform
        points_2d = ellip.squeeze(1)
        x_coords = points_2d[:, 0].astype(int)
        y_coords = points_2d[:, 1].astype(int)

        mask = np.zeros((w, h), dtype=bool)
        mask[x_coords, y_coords] = True
        mask = mask.T
        print("hough start")
        result = transform.hough_ellipse(mask, accuracy=5, threshold=10, min_size=int(w * 0.5), max_size=h)
        print("hough complete")
        result.sort(order='accumulator')
        best = list(result[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        orientation = best[5]
        center = (xc, yc)
        axis = (a, b)
        return center, axis, orientation
    else:
        return None, None, None


def detect_ellipses_cv2(image):
    """
    Recognized as "e-lsm" in detect.py
    e-lsm stands for enhanced lsm.
    In many cases, patterns on the hub are also included using edge detections. These pixels make ellipse detection more
    time-consuming and accuracy decreasing. Saving only the outer edge may help.
    Args:
        image (ndarray): image of tire.
    Return:
        fitted_params (tuple): fitted ellipse parameters.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, binary = cv2.threshold(magnitude_normalized, 20, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    for contour in contours:
        contour = interpolate_contour_gap(contour, max_gap=2)
        if cv2.contourArea(contour) < w*h*0.1 or cv2.contourArea(contour) > w*h*0.8:  # Tire should occupy a large area in cropped image. Ignore those too small.
            continue
        if len(contour) <= 8:
            continue

        ellipse = cv2.fitEllipse(contour)
        ellipses.append(ellipse)

    if len(ellipses) == 0:
        return None, None, None

    center, axis, orientation = ellipses[0]

    return center, axis, orientation


def detect_ellipses(image, ori_mode="e-lsm"):
    if ori_mode == "lsm":
        return detect_ellipses_lsm(image)
    if ori_mode == "e-lsm":
        return detect_ellipses_cv2(image)
    if ori_mode == "hough":
        return detect_ellipses_hough(image)
    if ori_mode == "e-hough":
        return detect_ellipses_enhanced_hough(image)

