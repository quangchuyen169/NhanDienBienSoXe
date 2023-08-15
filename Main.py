import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtGui import QPixmap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from giaodien import Ui_MainWindow


import DetectChars
import DetectPlates
import PossiblePlate

# Tùy chỉnh màu để nhận diện các hình ảnh vật thể
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True  # Bật True để hiện và sử dụng các bước trong DetectChars và DetectPlates

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.btn_img.clicked.connect(self.BrowserImg)
        self.uic.btn_img_detec.clicked.connect(self.main)
        self.text_edit = self.uic.txt_img
        self.linking = ""

    def BrowserImg(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        link, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        self.uic.img.setPixmap(QPixmap(link))
        self.linking = link

    def main(self):
        blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()

        if blnKNNTrainingSuccessful == False:
            print("\nerror: KNN training was not successful\n")
            return

        imgOriginalScene = cv2.imread(self.linking)

        if imgOriginalScene is None:
            print("\nerror: image not read from file\n\n")
            os.system("pause")
            return

        # Thay đổi kích thước ảnh nhập vào thành 800x600
        imgOriginalScene = cv2.resize(imgOriginalScene, (800, 600))

        listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)

        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")
        else:
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
            licPlate = listOfPossiblePlates[0]

            cv2.imshow("CROP PHAN BIEN SO XE", licPlate.imgPlate)
            #cv2.imshow("BIEN SO DA DUOC NHAN DIEN", licPlate.imgThresh)
            cv2.imwrite("output/CROP PHAN BIEN SO XE.png", licPlate.imgPlate)
            cv2.imwrite("output/BIEN SO DA DUOC NHAN DIEN.png", licPlate.imgThresh)

            if len(licPlate.strChars) == 0:
                print("\nno characters were detected\n\n")
                return
            self.drawGreenRectangleAroundPlate(imgOriginalScene, licPlate)

            cv2.imshow("ANH GOC VA DANH DAU PHAN NHAN DIEN", imgOriginalScene)
            cv2.imwrite("output/ANH GOC VA DANH DAU PHAN NHAN DIEN.png", imgOriginalScene)
            self.text_edit.append("Bien so xe la: " + licPlate.strChars)

        cv2.waitKey(0)
        return

    def drawGreenRectangleAroundPlate(self, imgOriginalScene, licPlate):
        p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene).astype(np.int32)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), tuple(SCALAR_GREEN), 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), tuple(SCALAR_GREEN), 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), tuple(SCALAR_GREEN), 2)
        cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), tuple(SCALAR_GREEN), 2)

if __name__ == "__main__":
    import sys
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
