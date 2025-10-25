import cv2
import numpy as np

url = "friren.jpg"

def print_img(url_get):

    img = cv2.imread(url_get, cv2.IMREAD_COLOR)

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("gr", cv2.WINDOW_NORMAL)

    img = np.asarray(img, np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 3)

    (n, m) = (3, 3)
    oper_Sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    oper_Sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    oper_Purit_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    oper_Purit_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    oper_x = oper_Sobel_x
    oper_y = oper_Sobel_y

    (h, w) = (len(img), len(img[0]))

    granic = np.zeros((h, w), np.uint8)
    
    matrix_len = np.zeros((h,w), np.float32)
    matrix_gr = np.zeros((h,w), np.float32)

    (h_s, w_s) = (int(n/2), int(m/2))

    max_len = 0

    for y in range(h):
        for x in range(w):
            G_x = 0
            G_y = 0
            for i in range(n):
                ind_y = y + i - h_s

                if(ind_y < 0):
                    ind_y = 0
                elif(ind_y >= h):
                    ind_y = h-1

                for j in range(m):
                    ind_x = x + j - w_s

                    if(ind_x < 0):
                        ind_x = 0
                    elif(ind_x >= w):
                        ind_x = w-1

                    G_x += int(img[ind_y][ind_x])*oper_x[i][j]
                    G_y += int(img[ind_y][ind_x])*oper_y[i][j]


            length = np.sqrt(int((G_x**2) + (G_y**2)))

            if(length > max_len):
                max_len = length

            matrix_len[y][x] = length

            gr = np.arctan(0)
            if(G_x != 0):
                gr = np.arctan(G_y/G_x)
            matrix_gr[y][x] = gr

    def napr(tang):
        if(tang < -2.414 or tang > 2.414):
            return [[0, 1, 0], [0, 0, 0], [0, 1, 0]]
        elif(tang < -0.414):
            return [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
        elif(tang > -0.414 and tang < 0.414):
            return [[0, 0, 0], [1, 0, 1], [0, 0, 0]]
        elif(tang < 2.414):
            return [[1, 0, 0], [0, 0, 0], [0, 0, 1]]
    print(matrix_len)
    print(matrix_gr)

    

    for y in range(h):
        for x in range(w):
            gran = True

            oper = napr(matrix_gr[y][x])

            for i in range(n):
                ind_y = y + i - h_s

                if(ind_y < 0):
                    ind_y = 0
                elif(ind_y >= h):
                    ind_y = h-1

                for j in range(m):
                    ind_x = x + j - w_s

                    if(ind_x < 0):
                        ind_x = 0
                    elif(ind_x >= w):
                        ind_x = w-1

                    if(oper[i][j] == 1):
                        if(matrix_len[ind_y][ind_x] >= matrix_len[y][x]):
                            gran = False
            
            if(gran):
                granic[y][x] = 255
    
    cv2.imshow("img", img)

    

    
    def change_level():

        low_level = cv2.getTrackbarPos('low_level', 'settings')
        high_level = cv2.getTrackbarPos('high_level', 'settings')

        granic_2 = np.array(granic)

        for y in range(h):
            for x in range(w):

                gran = False
                
                if(matrix_len[y][x] > high_level and matrix_len[y][x] < low_level):
                    for i in range(n):
                        ind_y = y + i - h_s

                        if(ind_y < 0):
                            ind_y = 0
                        elif(ind_y >= h):
                            ind_y = h-1

                        for j in range(m):
                            ind_x = x + j - w_s

                            if(ind_x < 0):
                                ind_x = 0
                            elif(ind_x >= w):
                                ind_x = w-1

                            if(ind_x != x or ind_y != y):
                                if(granic[ind_y][ind_x] == 255):
                                    gran = True
                elif(matrix_len[y][x] < low_level):
                    granic_2[y][x] = 0
                elif(matrix_len[y][x] > high_level):
                    gran = True
                
                if not(gran):
                    granic_2[y][x] = 0
        
        cv2.imshow("gr", granic_2)

    def nothing(*arg):
        pass

    def switch(value):
        change_level()

    cv2.imshow("gr", granic)

    cv2.namedWindow( "settings" )
    cv2.createTrackbar('high_level', 'settings', 10, int(max_len), switch)
    cv2.createTrackbar('low_level', 'settings', 25, int(max_len), switch)      

    #ker = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]],np.uint8)
    
    #granic = cv2.erode(granic, ker)
    cv2.waitKey(0)

def print_vid(url_video):

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.namedWindow("gr", cv2.WINDOW_NORMAL)

    vide_cap = cv2.VideoCapture(url_video)    

    (n, m) = (3, 3)
    oper_Sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    oper_Sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    w = int(vide_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vide_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    granic = np.zeros((h, w), np.uint8)
    
    matrix_len = np.zeros((h,w), np.float32)
    matrix_gr = np.zeros((h,w), np.float32)

    (h_s, w_s) = (int(n/2), int(m/2))

    def nothing(*arg):
        pass
    
    cv2.namedWindow( "settings" )
    cv2.createTrackbar('high_level', 'settings', 10, 255, nothing)
    cv2.createTrackbar('low_level', 'settings', 25, 255, nothing)

    def napr(tang):
        if(tang < -2.414 or tang > 2.414):
            return [[0, 1, 0], [0, 0, 0], [0, 1, 0]]
        elif(tang < -0.414):
            return [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
        elif(tang > -0.414 and tang < 0.414):
            return [[0, 0, 0], [1, 0, 1], [0, 0, 0]]
        elif(tang < 2.414):
            return [[1, 0, 0], [0, 0, 0], [0, 0, 1]]

    while True:

        ok, img = vide_cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (7,7), 3)

        cv2.imshow("img", img)

        max_len = 0

        for y in range(h):
            for x in range(w):
                G_x = 0
                G_y = 0
                for i in range(n):
                    ind_y = y + i - h_s

                    if(ind_y < 0):
                        ind_y = 0
                    elif(ind_y >= h):
                        ind_y = h-1

                    for j in range(m):
                        ind_x = x + j - w_s

                        if(ind_x < 0):
                            ind_x = 0
                        elif(ind_x >= w):
                            ind_x = w-1

                        G_x += int(img[ind_y][ind_x])*oper_Sobel_x[i][j]
                        G_y += int(img[ind_y][ind_x])*oper_Sobel_y[i][j]


                length = np.sqrt(int((G_x**2) + (G_y**2)))

                if(length > max_len):
                    max_len = length

                matrix_len[y][x] = length

                gr = np.arctan(0)
                if(G_x != 0):
                    gr = np.arctan(G_y/G_x)
                matrix_gr[y][x] = gr

        for y in range(h):
            for x in range(w):
                gran = True

                oper = napr(matrix_gr[y][x])

                for i in range(n):
                    ind_y = y + i - h_s

                    if(ind_y < 0):
                        ind_y = 0
                    elif(ind_y >= h):
                        ind_y = h-1

                    for j in range(m):
                        ind_x = x + j - w_s

                        if(ind_x < 0):
                            ind_x = 0
                        elif(ind_x >= w):
                            ind_x = w-1

                        if(oper[i][j] == 1):
                            if(matrix_len[ind_y][ind_x] >= matrix_len[y][x]):
                                gran = False
                
                if(gran):
                    granic[y][x] = 255  

        low_level = cv2.getTrackbarPos('low_level', 'settings')
        high_level = cv2.getTrackbarPos('high_level', 'settings')

        granic_2 = np.array(granic)

        for y in range(h):
            for x in range(w):

                gran = False
                
                if(matrix_len[y][x] > high_level and matrix_len[y][x] < low_level):
                    for i in range(n):
                        ind_y = y + i - h_s

                        if(ind_y < 0):
                            ind_y = 0
                        elif(ind_y >= h):
                            ind_y = h-1

                        for j in range(m):
                            ind_x = x + j - w_s

                            if(ind_x < 0):
                                ind_x = 0
                            elif(ind_x >= w):
                                ind_x = w-1

                            if(ind_x != x and ind_y != y):
                                if(granic_2[ind_y][ind_x] == 255):
                                    gran = True
                elif(matrix_len[y][x] < low_level):
                    granic[y][x] = 0
        
        cv2.imshow("gr", granic)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

print_img(url)
#print_vid(0)