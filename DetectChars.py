# DetectChars.py
import os

import cv2
import numpy as np
import math
import random
import sys

import Main
import Preprocess
import PossibleChar

# module level variables ##############################################################

kNearest = cv2.ml.KNearest_create() 
#Khởi tạo một bộ phân loại K-Nearest Neighbors (KNN) bằng cách sử dụng lớp cv2.ml.KNearest_create()
MIN_PIXEL_WIDTH = 2 #Độ rộng tối thiểu của 1 ký tự (tính bằng pixel)
MIN_PIXEL_HEIGHT = 8 #Chiều cao tối thiểu của 1 ký tự (tính bằng pixel)

MIN_ASPECT_RATIO = 0.25 #Tỷ lệ khung hình tối thiểu của một ký tự có thể (tính bằng tỷ lệ giữa chiều rộng và chiều cao).
MAX_ASPECT_RATIO = 1.0 #Tỷ lệ khung hình tối da

MIN_PIXEL_AREA = 80 #Diện tích tối thiểu của một ký tự có thể (tính bằng pixel).

        # constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3 # Khoảng cách tối thiểu (tính theo bội số) giữa hai đường chéo của hai ký tự có thể.
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0 # Khoảng cách tối đa (tính theo bội số) giữa hai đường chéo của hai ký tự có thể.

MAX_CHANGE_IN_AREA = 0.5 #Thay đổi tối đa của diện tích giữa các ký tự liền kề.

MAX_CHANGE_IN_WIDTH = 0.8 #Thay đổi tối đa của chiều rộng giữa các ký tự liền kề.
MAX_CHANGE_IN_HEIGHT = 0.2 #Thay đổi tối đa của chiều cao giữa các ký tự liền kề.

MAX_ANGLE_BETWEEN_CHARS = 12.0 #Góc tối đa giữa hai ký tự có thể.

        # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3 #Số ký tự tối thiểu để xem xét là một biển số hợp lệ.

RESIZED_CHAR_IMAGE_WIDTH = 20 #Chiều rộng của hình ảnh ký tự được thay đổi kích thước để phù hợp với mô hình nhận dạng.
RESIZED_CHAR_IMAGE_HEIGHT = 30 #Chiều cao của hình ảnh ký tự được thay đổi kích thước để phù hợp với mô hình nhận dạng.

MIN_CONTOUR_AREA = 100 #Diện tích tối thiểu của một contour (vùng được xác định bởi đường viền) để xem xét là một ký tự có thể.

###################################################################################################
def loadKNNDataAndTrainKNN():
    allContoursWithData = []     #Khởi tạo một danh sách rỗng để lưu trữ tất cả các contour (đường viền) có dữ liệu.
    validContoursWithData = []   #Khởi tạo một danh sách rỗng để lưu trữ các contour hợp lệ có dữ liệu.

    try: 
        npaClassifications = np.loadtxt("TranningData/classifications.txt", np.float32)
        #Đọc dữ liệu phân loại huấn luyện từ tệp "classifications.txt" và lưu vào mảng numpy npaClassifications.                 
    except:         #try gặp lỗi, chương trình sẽ nhảy vào khối except để xử lý lỗi.                                                                         
        print("error, unable to open classifications.txt, exiting program\n")  #thông báo lỗi cho người dùng, cho biết rằng không thể mở tệp "classifications.txt".
        os.system("pause") #tạm dừng chương trình và yêu cầu người dùng nhấn một phím để tiếp tục.
        return False       #Trả về giá trị False để chỉ ra rằng quá trình đọc dữ liệu phân loại đã thất bại.                                                               
    # end try

    try:
        npaFlattenedImages = np.loadtxt("TranningData/flattened_images.txt", np.float32)                
        #Đọc dữ liệu ảnh huấn luyện từ tệp "flattened_images.txt" và lưu vào mảng numpy npaFlattenedImages
    except:                                                                                 
        print("error, unable to open flattened_images.txt, exiting program\n")  
        os.system("pause")
        return False                                                                        
    # end try

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       
    #Thay đổi hình dạng của mảng npaClassifications thành một mảng 1 chiều, điều này là cần thiết để truyền cho hàm train()
    kNearest.setDefaultK(1)
    #Đặt giá trị K mặc định của bộ phân loại KNN thành 1.                                                             
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)           
    #Huấn luyện đối tượng KNN bằng cách truyền dữ liệu huấn luyện (npaFlattenedImages) và phân loại (npaClassifications) vào hàm train()
    
    return True                # Trả về giá trị True để chỉ ra rằng quá trình tải dữ liệu và huấn luyện đã thành công.

###################################################################################################
def detectCharsInPlates(listOfPossiblePlates):
    intPlateCounter = 0 #biến đếm biển số đã xử lý
    imgContours = None
    contours = []
    #Biến imgContours và contours được khởi tạo với giá trị ban đầu None và một danh sách rỗng, lưu trữ các đường viền (contour) được tìm thấy trong quá trình nhận dạng ký tự.
    if len(listOfPossiblePlates) == 0:          # nếu danh sách không có biển số nào
        return listOfPossiblePlates             # trả về ngay lập tức
    # end if

            # at this point we can be sure the list of possible plates has at least one plate

    for possiblePlate in listOfPossiblePlates:          
        #Vòng lặp này duyệt qua từng biển số ước tính trong danh sách listOfPossiblePlates.
        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)     # preprocess to get grayscale and threshold images
        # Hàm preprocess() trong module Preprocess được gọi để tiền xử lý ảnh biển số. Hàm này chuyển đổi ảnh biển số sang ảnh xám và thực hiện quá trình làm sạch ảnh bằng cách áp dụng các bước như làm mờ, làm rõ, chuyển đổi sang ảnh nhị phân.

        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx = 1.6, fy = 1.6)
        #Ảnh nhị phân của biển số được phóng to để dễ dàng quan sát và nhận dạng ký tự.
                
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #Hàm cv2.threshold() được sử dụng để áp dụng ngưỡng nhị phân lên ảnh biển số. Phương pháp Otsu được sử dụng để tự động xác định ngưỡng phân biệt giữa các pixel trắng và đen.
       
        
        listOfPossibleCharsInPlate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)
        #gọi hàm findPossibleCharsInPlate() để tìm kiếm các ký tự có thể có trong biển số. Đầu vào của hàm là ảnh xám (possiblePlate.imgGrayscale) và ảnh nhị phân (possiblePlate.imgThresh) của biển số.
        if Main.showSteps == True: # show steps ###################################################
            height, width, numChannels = possiblePlate.imgPlate.shape
            # Lấy kích thước chiều cao (height), chiều rộng (width) và số kênh (numChannels) của ảnh biển số (possiblePlate.imgPlate).
            imgContours = np.zeros((height, width, 3), np.uint8)
            #Tạo một mảng numpy có kích thước height x width x 3 và kiểu dữ liệu là uint8, đại diện cho ảnh đường viền.
            del contours[:]       # Xóa tất cả các phần tử trong danh sách contours (danh sách lưu trữ các đường viền).

            for possibleChar in listOfPossibleCharsInPlate:
                contours.append(possibleChar.contour)
            #Duyệt qua danh sách listOfPossibleCharsInPlate chứa các ký tự có thể có trong biển số và thêm đường viền của từng ký tự vào danh sách contours.

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            #Vẽ các đường viền trong danh sách contours lên ảnh imgContours bằng màu trắng.

        # end if # show steps #################################################################

        listOfListsOfMatchingCharsInPlate = findListOfListsOfMatchingChars(listOfPossibleCharsInPlate)
        #gọi hàm findListOfListsOfMatchingChars() để tìm kiếm danh sách các danh sách ký tự khớp trong biển số. Đầu vào của hàm là danh sách listOfPossibleCharsInPlate chứa các ký tự có thể có trong biển số.
        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            ##Tạo một mảng numpy có kích thước height x width x 3 và kiểu dữ liệu là uint8, đại diện cho ảnh đường viền.
            del contours[:]   # Xóa tất cả các phần tử trong danh sách contours (danh sách lưu trữ các đường viền).

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate:
                #Duyệt qua danh sách listOfListsOfMatchingCharsInPlate chứa các danh sách ký tự khớp trong biển số.
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)
            #Tạo ngẫu nhiên các giá trị màu RGB để đánh dấu các đường viền.

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # Duyệt qua từng ký tự khớp trong danh sách listOfMatchingChars và thêm đường viền của từng ký tự vào danh sách contours.
                # end for
                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
                #Vẽ các đường viền trong danh sách contours lên ảnh imgContours bằng màu ngẫu nhiên. Mỗi danh sách ký tự khớp sẽ có một màu đường viền riêng biệt để phân biệt các danh sách này.

        if (len(listOfListsOfMatchingCharsInPlate) == 0):           

           

            possiblePlate.strChars = ""
            continue                        
        #kiểm tra xem danh sách listOfListsOfMatchingCharsInPlate có rỗng không. Nếu danh sách này rỗng, có nghĩa là không tìm thấy bất kỳ danh sách ký tự khớp nào trong biển số. Trong trường hợp này, đoạn mã tiếp theo sẽ được thực hiện.

        for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  #duyệt qua từng danh sách listOfMatchingChars trong listOfListsOfMatchingCharsInPlate                        
            listOfListsOfMatchingCharsInPlate[i].sort(key = lambda matchingChar: matchingChar.intCenterX)   
            #Sắp xếp các ký tự trong danh sách listOfMatchingChars thứ i theo tọa độ intCenterX của mỗi ký tự. Điều này sẽ xếp các ký tự từ trái sang phải, giúp xác định thứ tự của các ký tự trong biển số.     
            listOfListsOfMatchingCharsInPlate[i] = removeInnerOverlappingChars(listOfListsOfMatchingCharsInPlate[i])             
        # Gọi hàm removeInnerOverlappingChars() để loại bỏ các ký tự chồng chéo trong danh sách listOfMatchingChars thứ i. Hàm này sẽ kiểm tra và loại bỏ các ký tự trùng lắp hoặc giao nhau trong biển số, đảm bảo rằng danh sách chỉ chứa các ký tự riêng biệt.

        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8)
            #Tạo một ảnh trắng đen (np.zeros) với kích thước (height, width, 3) để vẽ các đường viền. Đây là ảnh trắng đen với 3 kênh màu RGB.

            for listOfMatchingChars in listOfListsOfMatchingCharsInPlate: #Duyệt qua từng danh sách listOfMatchingChars trong listOfListsOfMatchingCharsInPlate.
                intRandomBlue = random.randint(0, 255)
                intRandomGreen = random.randint(0, 255)
                intRandomRed = random.randint(0, 255)
            #Tạo ngẫu nhiên giá trị màu RGB để sử dụng cho việc vẽ đường viền.
                del contours[:]
            # Xóa tất cả các phần tử trong danh sách contours để chuẩn bị vẽ các đường viền mới.
                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)
                # Duyệt qua từng ký tự khớp trong danh sách listOfMatchingChars và thêm đường viền của mỗi ký tự vào danh sách contours.

                cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))
            # Vẽ các đường viền trong danh sách contours lên ảnh imgContours với màu sắc xác định bởi giá trị ngẫu nhiên của intRandomBlue, intRandomGreen và intRandomRed.
        intLenOfLongestListOfChars = 0 # biến số lượng ký tự khớp nhiều nhất trong danh sách
        intIndexOfLongestListOfChars = 0 # biến chỉ số của danh sách có số lượng ký tự khớp nhiều nhất.
       
        for i in range(0, len(listOfListsOfMatchingCharsInPlate)): #duyệt qua từng danh sách trong listOfListsOfMatchingCharsInPlate.
            if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
                intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
                intIndexOfLongestListOfChars = i
    #kiểm tra xem số ký tự trong danh sách thứ i có lớn hơn intLenOfLongestListOfChars hay không. Nếu có, intLenOfLongestListOfChars được cập nhật với giá trị mới là len(listOfListsOfMatchingCharsInPlate[i]), và intIndexOfLongestListOfChars được gán bằng giá trị i.
            
        longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]
#gán giá trị của một danh sách con từ listOfListsOfMatchingCharsInPlate vào biến longestListOfMatchingCharsInPlate. Chỉ mục của danh sách con được chọn được xác định bởi biến intIndexOfLongestListOfChars. Điều này cho phép truy cập vào danh sách con dài nhất trong danh sách các danh sách con listOfListsOfMatchingCharsInPlate.
        if Main.showSteps == True: # show steps ###################################################
            imgContours = np.zeros((height, width, 3), np.uint8) #một ma trận hình ảnh imgContours được tạo với kích thước (height, width, 3) và kiểu dữ liệu np.uint8. Ma trận này được sử dụng để vẽ các đường viền.
            del contours[:] #danh sách contours được xóa để chuẩn bị cho việc thêm các đường viền mới.

            for matchingChar in longestListOfMatchingCharsInPlate: 
                contours.append(matchingChar.contour)
                #duyệt qua mỗi phần tử matchingChar trong danh sách longestListOfMatchingCharsInPlate. Đường viền của mỗi phần tử được thêm vào danh sách contours bằng cách sử dụng thuộc tính contour của đối tượng matchingChar.
        

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
#hàm cv2.drawContours được sử dụng để vẽ các đường viền từ danh sách contours lên ma trận hình ảnh imgContours. Giá trị -1 được truyền vào tham số thứ tư để vẽ tất cả các đường viền trong danh sách. Màu sắc của các đường viền được đặt thành màu trắng (Main.SCALAR_WHITE).

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longestListOfMatchingCharsInPlate)
#gán giá trị trả về từ hàm recognizeCharsInPlate vào thuộc tính strChars của đối tượng possiblePlate. Hàm recognizeCharsInPlate được gọi với hai đối số là possiblePlate.imgThresh ( ảnh nhị phân của biển số xe) và longestListOfMatchingCharsInPlate (danh sách các đường viền phù hợp nhất trong biển số xe).
        if Main.showSteps == True: # show steps ###################################################
           # print("chars found in plate number " + str(
                #intPlateCounter) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue . . .")
            #Thông báo bao gồm số thứ tự của biển số xe (intPlateCounter), chuỗi ký tự được nhận dạng từ biển số xe (possiblePlate.strChars) và một hướng dẫn cho người dùng.
            intPlateCounter = intPlateCounter + 1
            #Biến intPlateCounter được tăng lên 1 để chuẩn bị cho biển số xe tiếp theo. 
            cv2.waitKey(0)

    return listOfPossiblePlates
#trả về giá trị của biến listOfPossiblePlates.

########################################################################################
def findPossibleCharsInPlate(imgGrayscale, imgThresh):
    #Hàm findPossibleCharsInPlate nhận hai đối số: imgGrayscale (ảnh xám của biển số xe) và imgThresh (ảnh nhị phân của biển số xe). Nhiệm vụ của hàm này là tìm các ký tự tiềm năng trong biển số xe và trả về danh sách các ký tự tiềm năng đó.
    listOfPossibleChars = []       #tạo một danh sách rỗng listOfPossibleChars để chứa các ký tự tiềm năng. Đây sẽ là giá trị trả về của hàm.
    contours = []   #tạo một danh sách rỗng contours để chứa các đường viền tìm thấy.
    imgThreshCopy = imgThresh.copy() #Biến imgThreshCopy được khởi tạo bằng một bản sao của ảnh nhị phân imgThresh. Điều này được thực hiện để tránh làm thay đổi ảnh gốc trong quá trình tìm kiếm các đường viền.

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#Hàm cv2.findContours được sử dụng để tìm tất cả các đường viền trong ảnh nhị phân imgThreshCopy. Hai tham số cv2.RETR_LIST và cv2.CHAIN_APPROX_SIMPLE được sử dụng để chỉ định phương pháp lấy các đường viền và cấu trúc dữ liệu của các đường viền.
    for contour in contours: #duyệt qua từng đường viền trong danh sách contours                        
        possibleChar = PossibleChar.PossibleChar(contour) 
        #một đối tượng PossibleChar được tạo ra bằng cách truyền đường viền đó vào hàm khởi tạo của lớp PossibleChar.

        if checkIfPossibleChar(possibleChar):              
            listOfPossibleChars.append(possibleChar)       
#Hàm checkIfPossibleChar được gọi với đối số là possibleChar để kiểm tra xem đường viền đó có thể là ký tự tiềm năng hay không. Nếu điều kiện đúng, đối tượng possibleChar được thêm vào danh sách listOfPossibleChars.

    return listOfPossibleChars #danh sách listOfPossibleChars được trả về làm kết quả của hàm

########################################################################################
def checkIfPossibleChar(possibleChar):
#Hàm checkIfPossibleChar nhận một đối tượng possibleChar làm đối số và thực hiện một kiểm tra sơ bộ để xem liệu đối tượng đó có thể là một ký tự tiềm năng hay không.
    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
#Kiểm tra diện tích hình chữ nhật bao quanh ký tự (possibleChar.intBoundingRectArea) so với giá trị MIN_PIXEL_AREA.
#Kiểm tra chiều rộng hình chữ nhật bao quanh ký tự (possibleChar.intBoundingRectWidth) so với giá trị MIN_PIXEL_WIDTH.
#Kiểm tra chiều cao hình chữ nhật bao quanh ký tự (possibleChar.intBoundingRectHeight) so với giá trị MIN_PIXEL_HEIGHT.
#Kiểm tra tỷ lệ khung hình chữ nhật bao quanh ký tự (possibleChar.fltAspectRatio) nằm trong khoảng giá trị từ MIN_ASPECT_RATIO đến MAX_ASPECT_RATIO.
#Nếu tất cả các điều kiện trên được thoả mãn, tức là đối tượng possibleChar được xem là một ký tự tiềm năng và hàm trả về giá trị True. Ngược lại, nếu ít nhất một điều kiện không thoả mãn, hàm trả về giá trị False.

#######################################################################################
def findListOfListsOfMatchingChars(listOfPossibleChars):
#Hàm findListOfListsOfMatchingChars nhận một danh sách listOfPossibleChars làm đối số và thực hiện việc chia các ký tự tiềm năng thành các danh sách con của các ký tự khớp nhau.
    listOfListsOfMatchingChars = []      #Khởi tạo một danh sách rỗng listOfListsOfMatchingChars để chứa các danh sách con của các ký tự khớp nhau. Đây sẽ là giá trị trả về của hàm.

    for possibleChar in listOfPossibleChars:         #duyệt qua từng ký tự tiềm năng trong danh sách listOfPossibleChars
        listOfMatchingChars = findListOfMatchingChars(possibleChar, listOfPossibleChars)        
        #gọi hàm findListOfMatchingChars để tìm các ký tự trong danh sách listOfPossibleChars mà khớp với ký tự hiện tại.

        listOfMatchingChars.append(possibleChar)                
        #Các ký tự khớp được thêm vào danh sách listOfMatchingChars, bao gồm cả ký tự hiện tại. Điều này đảm bảo rằng danh sách này chứa tất cả các ký tự khớp với ký tự hiện tại.
        if len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS:    
            #Nếu danh sách không đủ số lượng (len(listOfMatchingChars) < MIN_NUMBER_OF_MATCHING_CHARS), vòng lặp sẽ tiếp tục và chuyển sang ký tự tiềm năng tiếp theo.
            continue                           
        
                                               
        listOfListsOfMatchingChars.append(listOfMatchingChars)      
#danh sách listOfMatchingChars, chứa các ký tự khớp với ký tự hiện tại, được thêm vào cuối danh sách listOfListsOfMatchingChars.
        listOfPossibleCharsWithCurrentMatchesRemoved = [] #một danh sách rỗng listOfPossibleCharsWithCurrentMatchesRemoved được khởi tạo

        listOfPossibleCharsWithCurrentMatchesRemoved = list(set(listOfPossibleChars) - set(listOfMatchingChars))
#loại bỏ các ký tự trong listOfMatchingChars khỏi listOfPossibleChars, dựa trên tính chất của tập hợp. Kết quả là danh sách listOfPossibleChars mới mà không chứa các ký tự đã được sử dụng trong listOfMatchingChars.
        recursiveListOfListsOfMatchingChars = findListOfListsOfMatchingChars(listOfPossibleCharsWithCurrentMatchesRemoved)      # recursive call
#gọi đệ quy findListOfListsOfMatchingChars được thực hiện với đối số là listOfPossibleCharsWithCurrentMatchesRemoved. Điều này cho phép tìm các danh sách con của các ký tự khớp nhau từ danh sách mới này. Kết quả của cuộc gọi đệ quy được gán vào recursiveListOfListsOfMatchingChars.
        for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:        #duyệt qua từng danh sách con recursiveListOfMatchingChars trong recursiveListOfListsOfMatchingChars
            listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)             # Mỗi danh sách con này được thêm vào listOfListsOfMatchingChars ban đầu bằng cách sử dụng phương thức append. Điều này đảm bảo rằng tất cả các danh sách con tìm thấy bởi cuộc gọi đệ quy cũng được thêm vào listOfListsOfMatchingChars.
        

        break       # thoát khỏi vòng lặp

    return listOfListsOfMatchingChars #listOfListsOfMatchingChars được trả về làm kết quả của hàm.

############################################################################################
def findListOfMatchingChars(possibleChar, listOfChars):
#hàm findListOfMatchingChars nhận đầu vào là possibleChar (ký tự có thể khớp) và listOfChars (danh sách các ký tự).           
    listOfMatchingChars = []                # khởi tạo rỗng, và nó sẽ được sử dụng để lưu trữ danh sách các ký tự khớp

    for possibleMatchingChar in listOfChars:         # duyệt qua từng ký tự possibleMatchingChar trong listOfChars.
        if possibleMatchingChar == possibleChar:    
            continue                                
    #nếu possibleMatchingChar trùng khớp với possibleChar, tức là nó là cùng một ký tự, thì chúng ta sẽ bỏ qua ký tự này bằng cách sử dụng lệnh continue. Việc này đảm bảo rằng ký tự hiện tại không được thêm vào danh sách ký tự khớp (listOfMatchingChars), để tránh sự trùng lặp trong danh sách.
        
        fltDistanceBetweenChars = distanceBetweenChars(possibleChar, possibleMatchingChar)
#fltDistanceBetweenChars là khoảng cách giữa hai ký tự possibleChar và possibleMatchingChar. Hàm distanceBetweenChars được sử dụng để tính toán giá trị này.
        fltAngleBetweenChars = angleBetweenChars(possibleChar, possibleMatchingChar)
#fltAngleBetweenChars là góc giữa hai ký tự possibleChar và possibleMatchingChar. Hàm angleBetweenChars được sử dụng để tính toán giá trị này.
        fltChangeInArea = float(abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(possibleChar.intBoundingRectArea)
#fltChangeInArea là tỷ lệ thay đổi diện tích giữa hai ký tự. Đây là giá trị tuyệt đối của hiệu giữa diện tích hình bao của possibleMatchingChar và possibleChar, chia cho diện tích của possibleChar. Điều này giúp đo lường sự khác biệt về diện tích giữa hai ký tự.
        fltChangeInWidth = float(abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(possibleChar.intBoundingRectWidth)
#fltChangeInWidth là tỷ lệ thay đổi chiều rộng giữa hai ký tự. Đây là giá trị tuyệt đối của hiệu giữa chiều rộng hình bao của possibleMatchingChar và possibleChar, chia cho chiều rộng của possibleChar. Điều này giúp đo lường sự khác biệt về chiều rộng giữa hai ký tự.
        fltChangeInHeight = float(abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(possibleChar.intBoundingRectHeight)
#fltChangeInHeight là tỷ lệ thay đổi chiều cao giữa hai ký tự. Đây là giá trị tuyệt đối của hiệu giữa chiều cao hình bao của possibleMatchingChar và possibleChar, chia cho chiều cao của possibleChar. Điều này giúp đo lường sự khác biệt về chiều cao giữa hai ký tự.
        if (fltDistanceBetweenChars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
            fltAngleBetweenChars < MAX_ANGLE_BETWEEN_CHARS and
            fltChangeInArea < MAX_CHANGE_IN_AREA and
            fltChangeInWidth < MAX_CHANGE_IN_WIDTH and
            fltChangeInHeight < MAX_CHANGE_IN_HEIGHT):
#kiểm tra khoảng cách giữa hai ký tự (fltDistanceBetweenChars) có nhỏ hơn một ngưỡng được tính dựa trên kích thước đường chéo của possibleChar (possibleChar.fltDiagonalSize) nhân với MAX_DIAG_SIZE_MULTIPLE_AWAY hay không. Nếu khoảng cách này nhỏ hơn ngưỡng, chúng ta tiếp tục kiểm tra các điều kiện khác.
#kiểm tra góc giữa hai ký tự (fltAngleBetweenChars) có nhỏ hơn MAX_ANGLE_BETWEEN_CHARS hay không.
#kiểm tra tỷ lệ thay đổi diện tích (fltChangeInArea), chiều rộng (fltChangeInWidth), và chiều cao (fltChangeInHeight) giữa hai ký tự có nhỏ hơn các ngưỡng tương ứng (MAX_CHANGE_IN_AREA, MAX_CHANGE_IN_WIDTH, MAX_CHANGE_IN_HEIGHT) hay không.
            listOfMatchingChars.append(possibleMatchingChar)        #Nếu tất cả các điều kiện trên đều đúng, tức là hai ký tự được coi là khớp nhau, chúng ta thêm possibleMatchingChar vào danh sách các ký tự khớp (listOfMatchingChars).

    return listOfMatchingChars                  # danh sách listOfMatchingChars chứa các ký tự khớp được trả về làm kết quả của hàm.

def distanceBetweenChars(firstChar, secondChar): #hàm distanceBetweenChars được sử dụng để tính toán khoảng cách Euclidean giữa hai ký tự firstChar và secondChar dựa trên tọa độ trung tâm của chúng.
    intX = abs(firstChar.intCenterX - secondChar.intCenterX) #độ chênh lệch theo trục X (intX) bằng cách lấy giá trị tuyệt đối của hiệu giữa firstChar.intCenterX và secondChar.intCenterX
    intY = abs(firstChar.intCenterY - secondChar.intCenterY) #chênh lệch theo trục Y (intY) bằng cách lấy giá trị tuyệt đối của hiệu giữa firstChar.intCenterY và secondChar.intCenterY.

    return math.sqrt((intX ** 2) + (intY ** 2)) #khoảng cách Euclidean bằng cách sử dụng công thức math.sqrt((intX ** 2) + (intY ** 2)), trong đó ** là toán tử mũ để tính bình phương.
#khoảng cách Euclidean giữa hai ký tự được trả về từ hàm.
def angleBetweenChars(firstChar, secondChar): #hàm angleBetweenChars được sử dụng để tính góc giữa hai ký tự firstChar và secondChar dựa trên tọa độ trung tâm của chúng.
    fltAdj = float(abs(firstChar.intCenterX - secondChar.intCenterX)) #tính toán độ chênh lệch theo trục X (fltAdj) bằng cách lấy giá trị tuyệt đối của hiệu giữa firstChar.intCenterX và secondChar.intCenterX
    fltOpp = float(abs(firstChar.intCenterY - secondChar.intCenterY)) #tính toán độ chênh lệch theo trục Y (fltOpp) bằng cách lấy giá trị tuyệt đối của hiệu giữa firstChar.intCenterY và secondChar.intCenterY.

    if fltAdj != 0.0:                          
        fltAngleInRad = math.atan(fltOpp / fltAdj)      
        #kiểm tra nếu fltAdj khác 0.0 để tránh chia cho 0. Nếu fltAdj khác 0.0, chúng ta tính toán góc trong radian (fltAngleInRad) bằng cách sử dụng hàm math.atan(fltOpp / fltAdj). Đây là góc tạo bởi cạnh kề (fltAdj) và cạnh đối (fltOpp) trong tam giác.
    else:
        fltAngleInRad = 1.5708                     #Nếu fltAdj bằng 0.0, chúng ta gán giá trị 1.5708 cho fltAngleInRad. Điều này đảm bảo rằng chương trình sẽ không gặp lỗi chia cho 0 trong trường hợp fltAdj bằng 0.0. Giá trị 1.5708 tương ứng với góc 90 độ.

    fltAngleInDeg = fltAngleInRad * (180.0 / math.pi)       #tính góc trong đơn vị độ (fltAngleInDeg) bằng cách nhân fltAngleInRad với 180.0 / math.pi. Điều này chuyển đổi góc từ radian sang độ.

    return fltAngleInDeg #Kết quả là góc giữa hai ký tự được trả về từ hàm, tính bằng đơn vị độ.

def removeInnerOverlappingChars(listOfMatchingChars): 
    #hàm removeInnerOverlappingChars được sử dụng để loại bỏ các ký tự chồng lấn bên trong trong danh sách listOfMatchingChars.
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)                
    #tạo một bản sao của danh sách listOfMatchingChars gán cho listOfMatchingCharsWithInnerCharRemoved. Điều này đảm bảo rằng chúng ta không thay đổi danh sách gốc mà chỉ làm việc với bản sao.

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
#duyệt qua từng ký tự trong danh sách listOfMatchingChars. Với mỗi ký tự hiện tại (currentChar), chúng ta kiểm tra với tất cả các ký tự khác (otherChar) trong danh sách.
            if currentChar != otherChar:        #Nếu currentChar và otherChar không giống nhau (khác nhau), chúng ta kiểm tra xem tâm của chúng có gần nhau không. 
                if distanceBetweenChars(currentChar, otherChar) < (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
# so sánh khoảng cách giữa chúng, tính bằng hàm distanceBetweenChars, với một ngưỡng (currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY).  
#Nếu khoảng cách giữa chúng nhỏ hơn ngưỡng, đó có nghĩa là chúng chồng lấn lẫn nhau. Chúng ta tiếp tục xác định ký tự nào nhỏ hơn, sau đó nếu ký tự đó chưa được loại bỏ trước đó, chúng ta loại bỏ nó.             
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:         
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:             
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)         
#Nếu currentChar có diện tích hình bao nhỏ hơn otherChar, và currentChar chưa được loại bỏ trước đó, chúng ta loại bỏ currentChar khỏi danh sách listOfMatchingCharsWithInnerCharRemoved                      
                    else:                                                                       # else if other char is smaller than current char
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:                # if other char was not already removed on a previous pass . . .
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)           # then remove other char
#nếu otherChar có diện tích hình bao nhỏ hơn currentChar, và otherChar chưa được loại bỏ trước đó, chúng ta loại bỏ otherChar.
    return listOfMatchingCharsWithInnerCharRemoved # danh sách listOfMatchingCharsWithInnerCharRemoved, chứa các ký tự đã loại bỏ các ký tự chồng lấn bên trong, được trả về từ hàm.

###################################################################################################
def recognizeCharsInPlate(imgThresh, listOfMatchingChars):
#hàm recognizeCharsInPlate được sử dụng để nhận dạng các ký tự trong biển số xe dựa trên danh sách listOfMatchingChars và hình ảnh ngưỡng imgThresh.
    strChars = ""   #khởi tạo biến strChars để lưu trữ kết quả nhận dạng ký tự. Đây là giá trị trả về của hàm.

    height, width = imgThresh.shape # lấy kích thước của hình ảnh imgThresh và gán giá trị chiều cao vào biến height và giá trị chiều rộng vào biến width.

    imgThreshColor = np.zeros((height, width, 3), np.uint8) 
    #tạo một hình ảnh màu imgThreshColor có kích thước là height x width và có 3 kênh màu (BGR), np.zeros() tạo một mảng numpy chứa các giá trị zero
    #tạo ra một mảng zero có kích thước (height, width, 3) và kiểu dữ liệu np.uint8, đại diện cho hình ảnh màu 8-bit.

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)       
#sử dụng để sắp xếp các ký tự trong listOfMatchingChars theo trục ngang từ trái qua phải. 
# Hàm sort() sắp xếp các phần tử trong danh sách dựa trên giá trị được trả về từ hàm lambda lambda matchingChar: matchingChar.intCenterX
# trong đó matchingChar là một phần tử trong danh sách và matchingChar.intCenterX là giá trị trục X của phần tử đó.

    cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR, imgThreshColor)                     # make color version of threshold image so we can draw contours in color on it
#chuyển đổi hình ảnh ngưỡng imgThresh từ không gian màu xám sang không gian màu RGB bằng cv2.cvtColor để chuẩn bị cho việc vẽ đường viền các ký tự màu sắc lên hình ảnh imgThreshColor
    
    for currentChar in listOfMatchingChars:   #lặp qua từng ký tự trong listOfMatchingChars.   
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY) 
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

       
#Điểm pt1 có tọa độ (currentChar.intBoundingRectX, currentChar.intBoundingRectY), tức là tọa độ x và y của góc trái trên của hình bao ký tự.
#Điểm pt2 có tọa độ ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth), (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight)), tức là tọa độ x và y của góc phải dưới của hình bao ký tự.
        cv2.rectangle(imgThreshColor, pt1, pt2, Main.SCALAR_GREEN, 2)          
#Vẽ hộp xung quanh ký tự lên hình ảnh imgThreshColor. Hộp được vẽ bằng màu xanh lá cây (Main.SCALAR_GREEN) với độ dày đường viền là 2.
          
        imgROI = imgThresh[currentChar.intBoundingRectY : currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                           currentChar.intBoundingRectX : currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]
# trích xuất một phần của hình ngưỡng imgThresh, bắt đầu từ vị trí hàng currentChar.intBoundingRectY đến hàng currentChar.intBoundingRectY + currentChar.intBoundingRectHeight, và từ vị trí cột currentChar.intBoundingRectX đến cột currentChar.intBoundingRectX + currentChar.intBoundingRectWidth. Kết quả là một hình ảnh ROI chỉ chứa ký tự hiện tại.
        imgROIResized = cv2.resize(imgROI, (RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))           # resize image, this is necessary for char recognition
#hình ảnh ROI imgROI được điều chỉnh kích thước bằng cv2.resize thành kích thước chuẩn RESIZED_CHAR_IMAGE_WIDTH và RESIZED_CHAR_IMAGE_HEIGHT. Việc điều chỉnh kích thước này cần thiết để phù hợp với việc nhận dạng ký tự.
        npaROIResized = imgROIResized.reshape((1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))        # flatten image into 1d numpy array
#điều chỉnh kích thước imgROIResized được chuyển đổi thành mảng 1D npaROIResized thông qua phương thức reshape. 
#imgROIResized là hình ảnh ký tự đã được thay đổi kích thước trước đó. Bằng cách sử dụng phương thức reshape() trên imgROIResized, ta chuyển đổi nó thành một mảng numpy có hình dạng (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT),
#  tức là mảng numpy với một hàng và số cột bằng tích của chiều rộng và chiều cao của ký tự đã thay đổi kích thước.
        npaROIResized = np.float32(npaROIResized)               #Mảng này sau đó được chuyển đổi từ kiểu dữ liệu nguyên (int) sang kiểu dữ liệu số thực (float) 
        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 1)              
# gọi phương thức findNearest() của đối tượng kNearest để thực hiện quá trình tìm ký tự gần nhất.
#  npaROIResized: Mảng numpy đã được thay đổi kích thước, đại diện cho ký tự cần nhận dạng.
#k=1: Tham số k xác định số lượng kết quả gần nhất cần trả về. Trong trường hợp này, ta chỉ cần kết quả gần nhất nên k được đặt là 1.
#retval: Giá trị trả về từ phương thức.
#npaResults: Mảng numpy chứa ký tự được dự đoán gần nhất.
#neigh_resp: Mảng numpy chứa các phản hồi lân cận liên quan đến ký tự được dự đoán.
#dists: Mảng numpy chứa khoảng cách giữa ký tự được dự đoán và các ký tự lân cận.

        strCurrentChar = str(chr(int(npaResults[0][0])))           
#chuyển đổi giá trị số nguyên trong mảng numpy npaResults thành một ký tự trong bảng mã Unicode.
#npaResults[0][0]: Truy cập phần tử đầu tiên của mảng numpy npaResults.
#int(npaResults[0][0]): Chuyển đổi giá trị số thực trong mảng numpy thành số nguyên.
#chr(int(npaResults[0][0])): Chuyển đổi số nguyên thành ký tự tương ứng trong bảng mã Unicode.
#str(chr(int(npaResults[0][0]))): Chuyển đổi ký tự thành chuỗi ký tự.
#Kết quả là một ký tự được lưu trong biến strCurrentChar, đại diện cho ký tự được dự đoán từ quá trình nhận dạng.

        strChars = strChars + strCurrentChar                    
    #ký tự hiện tại được thêm vào chuỗi strChars bằng cách sử dụng phép cộng chuỗi strChars = strChars + strCurrentChar.
    
    #if len(strChars) > 0:
       # output_string = "Bien so la: " + strChars
        #sys.stdout.write(output_string)
        #sys.stdout.flush()
    return strChars
#hàm trả về chuỗi strChars, chứa các ký tự đã được nhận dạng từ các ký tự trong mảng listOfMatchingChars.


