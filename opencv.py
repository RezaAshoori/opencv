import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=(255, 0, 0), thickness=3):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def bird_eye_view(image):
    height, width = image.shape[:2]
    
    # نقاط ورودی (نقاطی که باید به چشم پرنده تبدیل شوند)
    src_points = np.float32([[100, height], [width - 100, height], [width - 50, height // 2], [50, height // 2]])
    
    # نقاط خروجی (نقاطی که می‌خواهیم به آن‌ها تبدیل کنیم)
    dst_points = np.float32([[0, height], [width, height], [width, 0], [0, 0]])
    
    # محاسبه ماتریس تبدیل
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # اعمال تبدیل پرسپکتیو
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped_image

def process_image(image):
    # تبدیل به خاکستری
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # تشخیص لبه‌ها
    canny_image = cv2.Canny(gray_image, 100, 200)
    
    # تعریف ناحیه جاده
    height = image.shape[0]
    width = image.shape[1]
    roi_vertices = np.array([[100, height], [width - 100, height], [width / 2 + 50, height / 2], [width / 2 - 50, height / 2]], np.int32)
    
    # اعمال ناحیه جاده
    cropped_image = region_of_interest(canny_image, roi_vertices)
    
    # تشخیص خطوط
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=40, maxLineGap=25)
    
    # رسم خطوط در تصویر اصلی
    draw_lines(image, lines)
    
    # تبدیل به نمای چشم پرنده
    bird_eye_image = bird_eye_view(image)
    
    return bird_eye_image

# بارگذاری ویدیو یا تصویر
cap = cv2.VideoCapture('c:/Users/saba/Desktop/video.mp4')  # مسیر ویدیو را مشخص کنید

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = process_image(frame)
    
    cv2.imshow('Bird Eye View', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()