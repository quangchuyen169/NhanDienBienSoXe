# PossibleChar.py

import cv2
import numpy as np
import math

############################################################################################
class PossibleChar: #Lớp PossibleChar định nghĩa một đối tượng có thể là một ký tự tiềm năng trong quá trình nhận dạng biển số.

    # constructor #################################################################################
    def __init__(self, _contour): #hàm khởi tạo __init__ được sử dụng để khởi tạo một đối tượng PossibleChar với các thuộc tính sau:
        self.contour = _contour
#contour: Đường viền (contour) của ký tự. Đây là một đối tượng contour được truyền vào từ bên ngoài khi tạo một đối tượng PossibleChar.
        self.boundingRect = cv2.boundingRect(self.contour)
#boundingRect: Hình chữ nhật bao quanh ký tự. Hàm cv2.boundingRect được sử dụng để tính toán hình chữ nhật bao quanh contour và gán kết quả cho thuộc tính này.
        [intX, intY, intWidth, intHeight] = self.boundingRect

        self.intBoundingRectX = intX
        self.intBoundingRectY = intY
#Tọa độ x và y của góc trái trên của hình chữ nhật bao quanh ký tự.
        self.intBoundingRectWidth = intWidth
        self.intBoundingRectHeight = intHeight
#Chiều rộng và chiều cao của hình chữ nhật bao quanh ký tự.
        self.intBoundingRectArea = self.intBoundingRectWidth * self.intBoundingRectHeight
#Diện tích của hình chữ nhật bao quanh ký tự.
        self.intCenterX = (self.intBoundingRectX + self.intBoundingRectX + self.intBoundingRectWidth) / 2
        self.intCenterY = (self.intBoundingRectY + self.intBoundingRectY + self.intBoundingRectHeight) / 2
#Tọa độ x và y của điểm trung tâm của hình chữ nhật bao quanh ký tự.
        self.fltDiagonalSize = math.sqrt((self.intBoundingRectWidth ** 2) + (self.intBoundingRectHeight ** 2))
#Độ dài đường chéo của hình chữ nhật bao quanh ký tự
        self.fltAspectRatio = float(self.intBoundingRectWidth) / float(self.intBoundingRectHeight)
#Tỷ lệ giữa chiều rộng và chiều cao của hình chữ nhật bao quanh ký tự.