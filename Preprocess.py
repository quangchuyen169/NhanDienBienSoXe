# Preprocess.py

import cv2
import numpy as np
import math


# Các giá trị trong tiền xử lí Preprocess  ######################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
# kích thước của bộ lọc Gaussian được sử dụng trong quá trình làm mờ hình ảnh. Nó được xác định bằng cách chỉ định chiều rộng và chiều cao của bộ lọc, ví dụ (5, 5).
ADAPTIVE_THRESH_BLOCK_SIZE = 19
# kích thước khối sử dụng trong phương pháp ngưỡng thích ứng. Nó xác định kích thước của một khối hình vuông được sử dụng để tính giá trị ngưỡng cho mỗi điểm ảnh trong hình ảnh. Giá trị này phải là một số lẻ dương, ví dụ 19.
ADAPTIVE_THRESH_WEIGHT = 9
#hệ số trọng số trong phương pháp ngưỡng thích ứng. Nó được sử dụng để điều chỉnh mức độ ánh sáng trong quá trình xác định giá trị ngưỡng. Một giá trị lớn hơn sẽ tạo ra một giá trị ngưỡng lớn hơn cho các vùng tối hơn. Giá trị này cần là một số nguyên dương, ví dụ 9.

###################################################################################################
def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
#Hàm trích xuất kênh màu giá trị (value channel) từ ảnh gốc để chuyển đổi ảnh màu sang ảnh grayscale.
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)
#Hàm tăng cường độ tương phản của ảnh grayscale để làm nổi bật các đặc trưng.
    height, width = imgGrayscale.shape
#Lấy chiều cao và chiều rộng của ảnh grayscale.
    imgBlurred = np.zeros((height, width, 1), np.uint8)
#Khởi tạo một ảnh mờ có kích thước và kiểu dữ liệu tương tự như ảnh grayscale.
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
#Áp dụng bộ lọc Gaussian để làm mờ ảnh grayscale tăng cường độ tương phản. GAUSSIAN_SMOOTH_FILTER_SIZE là kích thước của bộ lọc Gaussian.
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
# Áp dụng phương pháp ngưỡng thích ứng để nhị phân hóa ảnh mờ. ADAPTIVE_THRESH_BLOCK_SIZE là kích thước khối sử dụng trong phương pháp ngưỡng thích ứng và ADAPTIVE_THRESH_WEIGHT là hệ số trọng số.
    return imgGrayscale, imgThresh
#Trả về ảnh grayscale và ảnh nhị phân sau khi tiền xử lý.
###################################################################################################
def extractValue(imgOriginal):
#Hàm extractValue thực hiện việc trích xuất kênh màu giá trị (value channel) từ ảnh màu. 
    height, width, numChannels = imgOriginal.shape
#Lấy chiều cao, chiều rộng và số kênh màu của ảnh gốc.
    imgHSV = np.zeros((height, width, 3), np.uint8)
#Khởi tạo một ảnh HSV (Hue, Saturation, Value) có kích thước tương tự với ảnh gốc.
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
#Chuyển đổi không gian màu từ BGR sang HSV. Hàm cv2.cvtColor được sử dụng để thực hiện việc chuyển đổi này.
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
#Tách kênh màu Hue, Saturation và Value từ ảnh HSV. Hàm cv2.split được sử dụng để tách các kênh màu này thành các ảnh riêng biệt.
    return imgValue
#Trả về ảnh kênh màu giá trị (value channel) sau khi trích xuất.

###################################################################################################
def maximizeContrast(imgGrayscale):
#Hàm maximizeContrast thực hiện việc tăng cường độ tương phản của ảnh grayscale. 
    height, width = imgGrayscale.shape
#Lấy chiều cao và chiều rộng của ảnh grayscale.
    imgTopHat = np.zeros((height, width, 1), np.uint8)
#Khởi tạo ảnh imgTopHat với kích thước và kiểu dữ liệu tương tự với ảnh grayscale. Ảnh này sẽ được sử dụng trong phép tophat.
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
# Khởi tạo ảnh imgBlackHat với kích thước và kiểu dữ liệu tương tự với ảnh grayscale. Ảnh này sẽ được sử dụng trong phép blackhat.
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#Tạo một phần tử cấu trúc (structuring element) có hình dạng hình chữ nhật kích thước (3, 3). Phần tử cấu trúc này sẽ được sử dụng trong các phép tophat và blackhat.
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
#Áp dụng phép tophat trên ảnh grayscale bằng cách sử dụng phép biến đổi hình thái học cv2.morphologyEx với lựa chọn cv2.MORPH_TOPHAT và phần tử cấu trúc đã tạo.
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)
#Áp dụng phép blackhat trên ảnh grayscale bằng cách sử dụng phép biến đổi hình thái học cv2.morphologyEx với lựa chọn cv2.MORPH_BLACKHAT và phần tử cấu trúc đã tạo.
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
#Tính tổng pixel-wise của ảnh grayscale ban đầu và ảnh tophat để tăng cường độ sáng các vùng sáng.
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
#Trừ ảnh blackhat từ ảnh grayscale cộng với tophat để tăng cường độ tối các vùng tối.
    return imgGrayscalePlusTopHatMinusBlackHat
# Trả về ảnh grayscale sau khi đã tăng cường độ tương phản.