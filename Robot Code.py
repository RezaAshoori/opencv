import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def draw_lines(img, lines, thickness=3):
    if lines is None:
        return

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue  # اجتناب از تقسیم بر صفر
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))

    def average_slope_intercept(lines):
        if len(lines) == 0:
            return None
        x_coords = []
        y_coords = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                x_coords += [x1, x2]
                y_coords += [y1, y2]
        if len(x_coords) == 0:
            return None
        m, b = np.polyfit(x_coords, y_coords, 1)
        return m, b

    left_fit = average_slope_intercept(left_lines)
    right_fit = average_slope_intercept(right_lines)

    y1 = img.shape[0]
    y2 = int(y1 * 0.6)

    if left_fit is not None:
        left_m, left_b = left_fit
        left_x1 = int((y1 - left_b) / left_m)
        left_x2 = int((y2 - left_b) / left_m)
        cv2.line(img, (left_x1, y1), (left_x2, y2), (0, 255, 0), thickness)

    if right_fit is not None:
        right_m, right_b = right_fit
        right_x1 = int((y1 - right_b) / right_m)
        right_x2 = int((y2 - right_b) / right_m)
        cv2.line(img, (right_x1, y1), (right_x2, y2), (255, 0, 0), thickness)

    if left_fit is not None and right_fit is not None:
        center_x1 = (left_x1 + right_x1) // 2
        center_x2 = (left_x2 + right_x2) // 2
        cv2.line(img, (center_x1, y1), (center_x2, y2), (0, 0, 255), thickness)

def process_frame(frame, prev_left_fit, prev_right_fit, alpha=0.7):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    vertices = np.array([[(50, frame.shape[0]), (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                          (frame.shape[1] // 2 + 50, frame.shape[0] // 2), (frame.shape[1] - 50, frame.shape[0])]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    line_img = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    new_left_fit, new_right_fit = None, None

    if lines is not None:
        left_lines = [line for line in lines if (line[0][2] - line[0][0]) != 0 and (line[0][1] - line[0][3]) / (line[0][2] - line[0][0]) < -0.5]
        right_lines = [line for line in lines if (line[0][2] - line[0][0]) != 0 and (line[0][1] - line[0][3]) / (line[0][2] - line[0][0]) > 0.5]

        if left_lines:
            new_left_fit = np.polyfit([y for line in left_lines for y in [line[0][1], line[0][3]]],
                                      [x for line in left_lines for x in [line[0][0], line[0][2]]], 1)
        if right_lines:
            new_right_fit = np.polyfit([y for line in right_lines for y in [line[0][1], line[0][3]]],
                                       [x for line in right_lines for x in [line[0][0], line[0][2]]], 1)

    if new_left_fit is not None:
        left_fit = (1 - alpha) * prev_left_fit + alpha * new_left_fit if prev_left_fit is not None else new_left_fit
        prev_left_fit = left_fit
    else:
        left_fit = prev_left_fit

    if new_right_fit is not None:
        right_fit = (1 - alpha) * prev_right_fit + alpha * new_right_fit if prev_right_fit is not None else new_right_fit
        prev_right_fit = right_fit
    else:
        right_fit = prev_right_fit

    if left_fit is not None and right_fit is not None:
        y1, y2 = frame.shape[0], int(frame.shape[0] * 0.6)
        left_x1, left_x2 = int((y1 - left_fit[1]) / left_fit[0]), int((y2 - left_fit[1]) / left_fit[0])
        right_x1, right_x2 = int((y1 - right_fit[1]) / right_fit[0]), int((y2 - right_fit[1]) / right_fit[0])

        cv2.line(line_img, (left_x1, y1), (left_x2, y2), (0, 255, 0), 3)
        cv2.line(line_img, (right_x1, y1), (right_x2, y2), (255, 0, 0), 3)

        center_x1, center_x2 = (left_x1 + right_x1) // 2, (left_x2 + right_x2) // 2
        cv2.line(line_img, (center_x1, y1), (center_x2, y2), (0, 0, 255), 3)

    return cv2.addWeighted(frame, 0.8, line_img, 1, 0), prev_left_fit, prev_right_fit

cap = cv2.VideoCapture('c:/Users/saba/Desktop/video.mp4')

fixed_width = 640
fixed_height = 480
prev_left_fit, prev_right_fit = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (fixed_width, fixed_height))
    processed_frame, prev_left_fit, prev_right_fit = process_frame(resized_frame, prev_left_fit, prev_right_fit)

    cv2.imshow('Resized Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
