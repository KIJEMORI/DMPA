import cv2
import numpy as np

img = cv2.imread("images.jpg")

def task1():

    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    ip_cam = cv2.VideoCapture(0)

    w = int(ip_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(ip_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while(True):
        ok, frame = ip_cam.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        #frame = cv2.inRange(frame, np.array([160,100,100]), np.array([255,0,0]))

        if not ok:
            break
        cv2.imshow("video", frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

#task1()

def tracker():
    cv2.namedWindow( "result_red" ) # создаем главное окно
    cv2.namedWindow("result_blue")
    cv2.namedWindow( "settings" ) # создаем окно настроек

    def nothing(*arg):
        pass

    img_red = cv2.imread("images.jpg")
    img_blue = cv2.imread("i.webp")

    # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    cv2.createTrackbar('R1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('G1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('B1', 'settings', 0, 255, nothing)
    cv2.createTrackbar('R2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('G2', 'settings', 255, 255, nothing)
    cv2.createTrackbar('B2', 'settings', 255, 255, nothing)
    crange = [0,0,0, 0,0,0]

    while True:
        
        #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV )

        # считываем значения бегунков
        R1 = cv2.getTrackbarPos('R1', 'settings')
        G1 = cv2.getTrackbarPos('G1', 'settings')
        B1 = cv2.getTrackbarPos('B1', 'settings')
        R2 = cv2.getTrackbarPos('R2', 'settings')
        G2 = cv2.getTrackbarPos('G2', 'settings')
        B2 = cv2.getTrackbarPos('B2', 'settings')

        # формируем начальный и конечный цвет фильтра
        h_min = np.array((R1, G1, B1), np.uint8)
        h_max = np.array((R2, G2, B2), np.uint8)

        # накладываем фильтр на кадр в модели HSV
        red = cv2.inRange(img_red, h_min, h_max)
        blue = cv2.inRange(img_blue, h_min, h_max)

        cv2.imshow('result_red', red) 
        cv2.imshow('result_blue', blue)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()



def task2():
    cv2.namedWindow("red on image", cv2.WINDOW_AUTOSIZE)

    

    img = cv2.inRange(img, np.array((0,0,73), np.uint8), np.array((59,80,255), np.uint8))

    cv2.imshow("red on image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#tracker()

#task2()

def task3():
    cv2.namedWindow("red on image erode", cv2.WINDOW_NORMAL)
    cv2.namedWindow("red on image dilate", cv2.WINDOW_NORMAL)

    

    img = cv2.inRange(img, np.array((0,0,73), np.uint8), np.array((59,80,255), np.uint8))

    img_erode = cv2.erode(img, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))
    #cv2.erode не подходит так как удаляет то что мне не нужно
    img_dilate = cv2.dilate(img, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))

    cv2.imshow("red on image erode", img_erode)
    cv2.imshow("red on image dilate", img_dilate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#task3()
def task4(img):
    
    img = cv2.inRange(img, np.array((0,0,73), np.uint8), np.array((59,80,255), np.uint8))
    
    img = cv2.dilate(img, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))

    array = np.asarray(img, np.bool)

    sum = np.sum(array)

    return sum
    
print(task4(img))

def task5():
    cv2.namedWindow("red on image", cv2.WINDOW_AUTOSIZE)

    stream = cv2.VideoCapture(0)

    # cv2.namedWindow( "settings" ) # создаем окно настроек
    # cv2.namedWindow("mask")

    # def nothing(*arg):
    #     pass

    # # создаем 6 бегунков для настройки начального и конечного цвета фильтра
    # cv2.createTrackbar('H1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('S1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('V1', 'settings', 0, 255, nothing)
    # cv2.createTrackbar('H2', 'settings', 255, 255, nothing)
    # cv2.createTrackbar('S2', 'settings', 255, 255, nothing)
    # cv2.createTrackbar('V2', 'settings', 255, 255, nothing)
    # crange = [0,0,0, 0,0,0]

    while stream.isOpened():
        ok, img = stream.read()

        if not ok:
            break

        # R1 = cv2.getTrackbarPos('H1', 'settings')
        # G1 = cv2.getTrackbarPos('S1', 'settings')
        # B1 = cv2.getTrackbarPos('V1', 'settings')
        # R2 = cv2.getTrackbarPos('H2', 'settings')
        # G2 = cv2.getTrackbarPos('S2', 'settings')
        # B2 = cv2.getTrackbarPos('V2', 'settings')

        #h_min = np.array((R1, G1, B1), np.uint16) # 160 87 196
        #h_max = np.array((R2, G2, B2), np.uint16) # 201 255 255

        h_min = np.array((160,87,196), np.uint8)
        h_max = np.array((201, 255, 255), np.uint8)
        #mask = cv2.inRange(img, np.array((0,0,73), np.uint8), np.array((59,80,255), np.uint8))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, h_min, h_max)
        mask = cv2.dilate(mask, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))
        #mask = cv2.erode(mask, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))
        #mask = cv2.erode(mask, np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8))

        #cv2.imshow("mask", mask)

        mask = np.asarray(mask, np.bool)

        x1 = 10000
        x2 = 0
        y1 = 10000
        y2 = 0

        for i_height in range(0,len(mask)):
            for j_width in range(0, len(mask)):
                if(mask[i_height][j_width]):
                    if(j_width > x2):
                        x2 = j_width
                    if(j_width < x1):
                        x1 = j_width
                    if(i_height > y2):
                        y2 = i_height
                    if(i_height<y1):
                        y1 = i_height
                    

        cv2.rectangle(img,(x1,y1), (x2,y2),(0,0,0), 2)

    #оставь это img = cv2.bitwise_not(img, mask)

    

    

        cv2.imshow("red on image",img)
        if(cv2.waitKey(1) & 0xFF == 27):
            break
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

task5()
