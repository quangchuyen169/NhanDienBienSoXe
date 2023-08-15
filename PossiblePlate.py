# PossiblePlate.py

import cv2
import numpy as np

############################################################################################
class PossiblePlate:    #Lớp PossiblePlate đại diện cho một biển số tiềm năng trong quá trình nhận dạng biển số. 
    
    def __init__(self):
#Phương thức __init__ là phương thức khởi tạo của lớp PossiblePlate, được sử dụng để khởi tạo các thuộc tính ban đầu của đối tượng PossiblePlate. Các thuộc tính hình ảnh và chuỗi ký tự được khởi tạo ban đầu là None và rỗng ("") tương ứng.
        self.imgPlate = None
        self.imgGrayscale = None
        self.imgThresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""
#imgPlate: Hình ảnh của biển số đã được cắt ra từ hình ảnh gốc.
#imgGrayscale: Hình ảnh biển số được chuyển đổi sang không gian màu xám.
#imgThresh: Hình ảnh biển số đã được xử lý ngưỡng (threshold).
#rrLocationOfPlateInScene: Đối tượng RotatedRect (hình chữ nhật xoay) đại diện cho vị trí của biển số trong hình ảnh gốc.
#strChars: Chuỗi các ký tự được nhận dạng từ biển số.