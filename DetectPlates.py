# DetectPlates.py

import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
#các hệ số đệm cho chiều rộng và chiều cao của biển số xe. Các hệ số này sẽ được sử dụng để tăng kích thước của vùng chứa biển số trước khi nhận diện, nhằm đảm bảo rằng không bỏ sót các phần quan trọng trong biển số.
##########################################################################
def detectPlatesInScene(imgOriginalScene): #Hàm detectPlatesInScene nhận đầu vào là imgOriginalScene, tức là hình ảnh gốc chứa biển số xe. Hàm này sẽ trả về một danh sách các biển số có thể tìm thấy trong hình ảnh đó.
    listOfPossiblePlates = []     #  khởi tạo một danh sách rỗng để lưu trữ các biển số có thể.            

    height, width, numChannels = imgOriginalScene.shape 
    #lấy kích thước của hình ảnh gốc và gán cho các biến height, width, và numChannels. Biến height là chiều cao của hình ảnh, width là chiều rộng và numChannels là số kênh màu của hình ảnh.

    imgGrayscaleScene = np.zeros((height, width, 1), np.uint8)
    imgThreshScene = np.zeros((height, width, 1), np.uint8)
    #imgGrayscaleScene và imgThreshScene là các mảng 2 chiều có kích thước tương tự với hình ảnh gốc, được khởi tạo với các giá trị 0 và kiểu dữ liệu np.uint8 (8-bit không dấu). 
    imgContours = np.zeros((height, width, 3), np.uint8)
    #imgContours là một mảng 3 chiều, cũng có kích thước tương tự với hình ảnh gốc, được khởi tạo với các giá trị 0 và kiểu dữ liệu np.uint8. Mảng này được sử dụng để vẽ các đường viền trên hình ảnh.
    cv2.destroyAllWindows() #đóng tất cả các cửa sổ đồ họa hiện tại nếu có.

    if Main.showSteps == True: #nếu biến showSteps trong lớp Main có giá trị là True . 
        cv2.imshow("ANH GOC DE NHAN DIEN", imgOriginalScene)
    #hình ảnh gốc imgOriginalScene sẽ được hiển thị trong cửa sổ với tiêu đề "ANH GOC DE NHAN DIEN" bằng hàm cv2.imshow().
 

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(imgOriginalScene)        
    #gọi hàm preprocess từ module Preprocess và truyền imgOriginalScene vào. Hàm preprocess được sử dụng để xử lý hình ảnh gốc và trả về một cặp giá trị là imgGrayscaleScene (hình ảnh xám) và imgThreshScene (hình ảnh ngưỡng).

    if Main.showSteps == True: # show steps #######################################################
        #cv2.imshow("CHUYEN ANH MAU QUA XAM", imgGrayscaleScene)
        cv2.imwrite("output/CHUYEN ANH MAU QUA XAM.png", imgGrayscaleScene)
#Hình ảnh xám imgGrayscaleScene được hiển thị với tiêu đề "CHUYEN ANH MAU QUA XAM" và được lưu vào file "output/CHUYEN ANH MAU QUA XAM.png" bằng hàm cv2.imshow() và cv2.imwrite()        
        #cv2.imshow("NGUOC SANG ANH VA DO TUONG PHAN CAO", imgThreshScene)
        cv2.imwrite("output/NGUOC SANG ANH VA DO TUONG PHAN CAO.png", imgGrayscaleScene)
# hình ảnh ngưỡng imgThreshScene được hiển thị với tiêu đề "NGUOC SANG ANH VA DO TUONG PHAN CAO" và được lưu vào file "output/NGUOC SANG ANH VA DO TUONG PHAN CAO.png".
    
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene)
#gọi hàm findPossibleCharsInScene và truyền imgThreshScene vào để tìm các ký tự có thể trong hình ảnh ngưỡng. Kết quả được gán cho biến listOfPossibleCharsInScene.
    if Main.showSteps == True: #nếu biến showSteps trong lớp Main có giá trị là True .
        #print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            #len(listOfPossibleCharsInScene)))
        #in ra số lượng các ký tự có thể trong danh sách listOfPossibleCharsInScene

        imgContours = np.zeros((height, width, 3), np.uint8)
    #khởi tạo một mảng numpy imgContours với kích thước tương tự với hình ảnh gốc, có kiểu dữ liệu np.uint8 và được gán giá trị 0. Mảng này sẽ được sử dụng để vẽ các đường viền.
        contours = [] #khởi tạo một danh sách rỗng để lưu trữ các đường viền.

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
#mỗi phần tử possibleChar trong danh sách listOfPossibleCharsInScene được lấy đường viền (contour) của nó và thêm vào danh sách contours.

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
#vẽ các đường viền lên hình ảnh imgContours sử dụng hàm cv2.drawContours(). Main.SCALAR_WHITE là một hằng số đại diện cho màu trắng.
        #cv2.imshow("LAY KI TU CO THE LA BIEN SO", imgContours)
#hiển thị hình ảnh imgContours trong cửa sổ với tiêu đề "LAY KI TU CO THE LA BIEN SO" bằng hàm cv2.imshow().
        cv2.imwrite("output/LAY KI TU CO THE LA BIEN SO.png", imgContours)
#lưu hình ảnh imgContours vào file "output/LAY KI TU CO THE LA BIEN SO.png" bằng hàm cv2.imwrite().

    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)
# gọi hàm findListOfListsOfMatchingChars từ module DetectChars và truyền danh sách listOfPossibleCharsInScene vào để tìm các danh sách ký tự khớp nhau. Kết quả được gán vào biến listOfListsOfMatchingCharsInScene.
    if Main.showSteps == True: #nếu biến showSteps trong lớp Main có giá trị là True .
        #print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
           # len(listOfListsOfMatchingCharsInScene)))  
#in ra số lượng danh sách các ký tự khớp nhau trong listOfListsOfMatchingCharsInScene.

        imgContours = np.zeros((height, width, 3), np.uint8)
#khởi tạo một mảng numpy imgContours với kích thước tương tự với hình ảnh gốc, có kiểu dữ liệu np.uint8 và được gán giá trị 0. Mảng này sẽ được sử dụng để vẽ các đường viền.
        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
#mỗi danh sách listOfMatchingChars trong listOfListsOfMatchingCharsInScene được lặp qua. Điều này đại diện cho việc tìm thấy một nhóm các ký tự khớp nhau.
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []
#Trong vòng lặp này, một giá trị màu ngẫu nhiên được tạo ra để sử dụng trong việc vẽ đường viền. Đường viền của mỗi ký tự trong danh sách listOfMatchingChars được lấy và thêm vào danh sách contours.
            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
# mỗi phần tử matchingChar trong danh sách listOfMatchingChars (danh sách các ký tự khớp nhau) được lặp qua. 
# Đường viền (contour) của mỗi ký tự trong danh sách này được lấy và thêm vào danh sách contours 

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
#vẽ các đường viền trong danh sách contours lên hình ảnh imgContours màu sắc của các đường viền được xác định bằng cách sử dụng giá trị màu ngẫu nhiên tương ứng từ các biến intRandomBlue, intRandomGreen, và intRandomRed.

       # cv2.imshow("KI TU BIEN SO DUOC NHAN DIEN", imgContours)
        cv2.imwrite("output/KI TU BIEN SO DUOC NHAN DIEN.png", imgContours)
#hình ảnh imgContours được hiển thị trong cửa sổ có tiêu đề "KI TU BIEN SO DUOC NHAN DIEN" bằng hàm cv2.imshow(), và cũng được lưu vào file "output/KI TU BIEN SO DUOC NHAN DIEN.png" bằng hàm cv2.imwrite().

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:           #lặp qua từng danh sách các ký tự khớp nhau trong listOfListsOfMatchingCharsInScene.
        possiblePlate = extractPlate(imgOriginalScene, listOfMatchingChars)         
#hàm extractPlate() được gọi để trích xuất biển số từ hình ảnh gốc imgOriginalScene và danh sách các ký tự khớp nhau listOfMatchingChars. Kết quả trả về là một đối tượng possiblePlate chứa thông tin về biển số.
        if possiblePlate.imgPlate is not None:           
            listOfPossiblePlates.append(possiblePlate)                 
#Nếu biển số được tìm thấy (tức là imgPlate trong possiblePlate không phải là None), thì biển số đó được thêm vào danh sách listOfPossiblePlates thông qua lệnh listOfPossiblePlates.append(possiblePlate).

        #print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)
#in ra thông báo hoàn tất quá trình nhận dạng biển số và yêu cầu người dùng nhấn một phím để bắt đầu quá trình nhận dạng các ký tự.

    return listOfPossiblePlates
#danh sách listOfPossiblePlates chứa các biển số có thể được tìm thấy trong hình ảnh được trả về từ hàm.

##########################################################################################
def findPossibleCharsInScene(imgThresh): #hàm findPossibleCharsInScene nhận đầu vào là hình ảnh nhị phân imgThresh, đại diện cho một hình ảnh đã qua xử lý ngưỡng.
    listOfPossibleChars = []      #Khởi tạo  danh sách rỗng, chứa các ký tự có thể, sẽ được trả về là kết quả cuối cùng của hàm.

    intCountOfPossibleChars = 0 #biến đếm số lượng các ký tự có thể.

    imgThreshCopy = imgThresh.copy() #một bản sao của hình ảnh nhị phân imgThresh, được sử dụng để thực hiện các thao tác trên bản sao và giữ nguyên hình ảnh gốc.

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
    #contours và npaHierarchy được trả về từ hàm cv2.findContours(), được sử dụng để tìm tất cả các đường viền trong hình ảnh nhị phân imgThreshCopy.
#cv2.RETR_LIST, chỉ định phương pháp lấy các đường viền. Trong trường hợp này, tất cả các đường viền đều được lưu trữ và không có cấu trúc phân cấp.
#cv2.CHAIN_APPROX_SIMPLE, chỉ định phương pháp xấp xỉ đường viền. Phương pháp này xấp xỉ các đường viền bằng cách loại bỏ các điểm trung gian và chỉ lưu trữ các đỉnh quan trọng để mô tả hình dạng đường viền.
    height, width = imgThresh.shape #height và width lấy kích thước của hình ảnh imgThresh.
    imgContours = np.zeros((height, width, 3), np.uint8) #
#một hình ảnh trống imgContours được tạo ra có kích thước (height, width, 3) và kiểu dữ liệu np.uint8. Hình ảnh này sẽ được sử dụng để vẽ các đường viền.
    for i in range(0, len(contours)):          #duyệt qua tất cả các đường viền trong danh sách contours

        if Main.showSteps == True: #nếu biến showSteps trong lớp Main có giá trị là True .
            cv2.drawContours(imgContours, contours, i, Main.SCALAR_WHITE)
#đường viền đó sẽ được vẽ trên hình ảnh imgContours bằng cách sử dụng hàm cv2.drawContours(). Điều này giúp hiển thị các bước trong quá trình tìm kiếm các ký tự (nếu được bật).

        possibleChar = PossibleChar.PossibleChar(contours[i])
#Một đối tượng PossibleChar mới được tạo với đường viền là đường viền hiện tại trong vòng lặp.
        if DetectChars.checkIfPossibleChar(possibleChar):    #Nếu đường viền được xem là một ký tự có thể (thông qua kiểm tra trong hàm checkIfPossibleChar() của lớp DetectChars)               
            intCountOfPossibleChars = intCountOfPossibleChars + 1    #biến intCountOfPossibleChars được tăng lên 1      
            listOfPossibleChars.append(possibleChar)    #đối tượng possibleChar được thêm vào danh sách listOfPossibleChars.                    


    if Main.showSteps == True: #nếu biến showSteps trong lớp Main có giá trị là True .
        #print("\nstep 2 - len(contours) = " + str(len(contours)))  
        #print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  
        #cv2.imshow("LAM NOI BAT CAC CHI TIET CO THE LA KI TU", imgContours)
        cv2.imwrite("output/LAM NOI BAT CAC CHI TIET CO THE LA KI TU.png", imgContours)
    
#các thông tin sau sẽ được in ra màn hình:
#Số lượng đường viền trong danh sách contours thông qua câu lệnh print("step 2 - len(contours) = " + str(len(contours))).
#Số lượng ký tự có thể được tìm thấy trong hình ảnh thông qua câu lệnh print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars)).
#Hình ảnh imgContours sẽ được hiển thị trong cửa sổ với tiêu đề "LAM NOI BAT CAC CHI TIET CO THE LA KI TU" bằng hàm cv2.imshow().
#Hình ảnh imgContours cũng được lưu vào tệp tin "output/LAM NOI BAT CAC CHI TIET CO THE LA KI TU.png" bằng hàm cv2.imwrite().
    return listOfPossibleChars #danh sách listOfPossibleChars chứa các đối tượng PossibleChar sẽ được trả về từ hàm.
###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
#Hàm extractPlate nhận đầu vào là hình ảnh gốc imgOriginal và danh sách listOfMatchingChars chứa các ký tự trùng khớp.
    possiblePlate = PossiblePlate.PossiblePlate()           
#Một đối tượng PossiblePlate mới được tạo và gán cho biến possiblePlate. Đối tượng này sẽ là giá trị trả về của hàm, đại diện cho biển số có thể được trích xuất từ hình ảnh.

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position
#Danh sách listOfMatchingChars được sắp xếp dựa trên vị trí trục x (intCenterX) của các ký tự. Việc này giúp sắp xếp các ký tự từ trái sang phải trên biển số.
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY
#Điểm trung tâm của biển số được tính toán bằng cách lấy trung bình của intCenterX và intCenterY của ký tự đầu tiên và ký tự cuối cùng trong danh sách. Kết quả được lưu vào fltPlateCenterX và fltPlateCenterY.
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)
#intPlateWidth là chiều rộng dự đoán của biển số, được tính toán bằng cách lấy khoảng cách từ vị trí x của ký tự cuối cùng trừ đi vị trí x của ký tự đầu tiên, sau đó nhân với hệ số PLATE_WIDTH_PADDING_FACTOR. Hệ số này giúp tăng kích thước dự đoán của biển số so với khoảng cách thực tế giữa các ký tự.
    intTotalOfCharHeights = 0
#khởi tạo intTotalOfCharHeights là tổng chiều cao của tất cả các ký tự trong danh sách listOfMatchingChars.
    for matchingChar in listOfMatchingChars: #lặp qua từng ký tự trong danh sách listOfMatchingChars
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    #intTotalOfCharHeights được tính toán bằng cách cộng dồn chiều cao của từng ký tự trong listOfMatchingChars

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)
#fltAverageCharHeight là chiều cao trung bình của các ký tự trong danh sách. Nó được tính bằng cách chia tổng chiều cao của các ký tự chia cho số lượng ký tự trong danh sách.
    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)
#intPlateHeight là chiều cao dự đoán của biển số, được tính bằng cách nhân chiều cao trung bình của các ký tự với hệ số PLATE_HEIGHT_PADDING_FACTOR. Hệ số này giúp tăng kích thước dự đoán của biển số so với chiều cao trung bình của các ký tự.
            
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
#fltOpposite là hiệu giữa tọa độ Y của ký tự cuối cùng trong danh sách và ký tự đầu tiên trong danh sách.
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
#fltHypotenuse là khoảng cách giữa hai ký tự đầu tiên và cuối cùng trong danh sách, được tính bằng hàm distanceBetweenChars
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
#fltCorrectionAngleInRad là góc sửa đổi (sai số) tính bằng cách tính asin của fltOpposite chia cho fltHypotenuse.
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)
#fltCorrectionAngleInDeg là góc sửa đổi được chuyển đổi từ radian sang độ.
            
    possiblePlate.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )
#Các thông số của vùng biển số bao gồm tọa độ trung tâm, chiều rộng và chiều cao, cùng với góc sửa đổi, được gói vào biến thành viên rrLocationOfPlateInScene của đối tượng possiblePlate.
          
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)
#rotationMatrix là ma trận xoay được tạo bằng cách sử dụng cv2.getRotationMatrix2D với các tham số tọa độ trung tâm ptPlateCenter, góc sửa đổi fltCorrectionAngleInDeg, và tỷ lệ mở rộng 1.0.
    height, width, numChannels = imgOriginal.shape      # unpack original image width and height
#height, width, numChannels là các thông số chiều cao, chiều rộng và số kênh màu của hình ảnh gốc được giải nén.
    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image
#imgRotated là hình ảnh gốc được xoay bằng cv2.warpAffine sử dụng rotationMatrix.
    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))
#imgCropped là vùng được cắt từ imgRotated sử dụng cv2.getRectSubPix với kích thước của biển số (intPlateWidth và intPlateHeight) và tọa độ trung tâm ptPlateCenter.
    possiblePlate.imgPlate = imgCropped         
#hình ảnh đã cắt được gán vào biến thành viên imgPlate của đối tượng possiblePlate.

    return possiblePlate
# đối tượng possiblePlate chứa hình ảnh biển số được cắt và các thông tin liên quan sẽ được trả về từ hàm.