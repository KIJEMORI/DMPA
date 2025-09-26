import cv2
import numpy as np

url_img = "C:\\Users\\KIJEMORI\\Downloads\\friren.jpg"

img = cv2.imread(url_img, cv2.IMREAD_GRAYSCALE)

#print(img)

def show_image():
    cv2.namedWindow("WND", cv2.WINDOW_NORMAL)
    cv2.imshow("WND", img)
    cv2.waitKey(0)

    url_img = "C:\\Users\\KIJEMORI\\Downloads\\chest_close.png"

    img = cv2.imread(url_img, cv2.IMREAD_ANYCOLOR)

    cv2.namedWindow("WND", cv2.WINDOW_GUI_EXPANDED)
    cv2.imshow("WND", img)
    cv2.waitKey(0)


    url_img = "C:\\Users\\KIJEMORI\\Downloads\\TJm1gZ9mvKMNhzA3ZNSSZ4X78u6O7ixwzCveaBUr.webp"

    img = cv2.imread(url_img, cv2.IMREAD_GRAYSCALE)

    cv2.namedWindow("WND", cv2.WINDOW_FULLSCREEN)
    cv2.imshow("WND", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

#show_image()

url_video = "C:\\Users\\KIJEMORI\\Desktop\\reze-chainsaw-man.mp4"

def play_video():

    video = cv2.VideoCapture(url_video, cv2.CAP_ANY)

    cv2.namedWindow("Video", cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("Video-Resize", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Video-recolor", cv2.WINDOW_NORMAL)

    while video.isOpened():
        ret, frame = video.read()

        if not(ret):
            break

        cv2.imshow("Video", frame)

        resize = cv2.resize(frame, (500, 500))
        cv2.imshow("Video-Resize", resize)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Video-recolor", gray)

        if(cv2.waitKey(50) & 0xFF == 27):
            break

    video.release()
    cv2.destroyAllWindows()

def copy_video():
    video = cv2.VideoCapture(url_video, cv2.CAP_ANY)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_output = cv2.VideoWriter("lolo.mov", fourcc, 25, (w,h))

    while video.isOpened():
        ret, frame = video.read()

        if not(ret):
            break

        video_output.write(frame)

        if(cv2.waitKey(50) & 0xFF == 27):
            break


    video.release()
    video_output.release()
    cv2.destroyAllWindows()

def to_HSV_format():
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.imshow("RGB", img)


    cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV", hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows() 



def rect_on_image():
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)

    url_ip_cam = "rtsp://192.168.39.179:8080/h264_ulaw.sdp"

    stream = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    w = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_output = cv2.VideoWriter("lolo.mov", fourcc, 25, (w,h))

    while stream.isOpened():
        ret, frame = stream.read()

        if not(ret):
            break
        
        #cv2.rectangle(frame, (int(w/2-50), int(h/2-100)), (int(w/2+50), int(h/2+100)), (0, 0, 255), 5)
        #cv2.rectangle(frame, (int(w/2-100), int(h/2-50)), (int(w/2+100), int(h/2+50)), (0, 0, 255), 5)

        cv2.circle(frame, (int(w/2), int(h/2)), 250, (0, 0, 255), 5)

        cv2.imshow("img", frame)

        if(cv2.waitKey(1) & 0xFF == 27):
            break

    #cv2.waitKey(0)
    cv2.destroyAllWindows()

def readIPCam():
    cv2.namedWindow("img", cv2.WINDOW_AUTOSIZE)

    video = cv2.VideoCapture("https://192.168.39.179:8080/stream1")

    
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output.mov", fourcc, 25, (w,h))
    while (True):
        ok,img = video.read()
        if not ok:
            break

        cv2.imshow('img', img)
        video_writer.write(img)
        if( cv2.waitKey(1) & 0xFF == ord('q')):
            break

    video.release()
    cv2.destroyAllWindows()
#rect_on_image()
#readIPCam()
# to_HSV_format()

def write_video_ip_cam():
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    ip_cam = cv2.VideoCapture("rtsp://192.168.170.77:8080/h264_ulaw.sdp")

    w = int(ip_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(ip_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter("ip_cam_output.mp4", fourcc, 25, (w,h))

    while(True):
        ok, img = ip_cam.read()
        if not ok:
            break
        cv2.imshow("video", img)
        video_writer.write(img)

        if(cv2.waitKey(24) & 0xFF == ord('q')):
            break

    video_writer.release()
    cv2.destroyAllWindows()

def get_video_and_get_RGB_color():

    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)

    ip_cam = cv2.VideoCapture(0)

    w = int(ip_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(ip_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ind_pix_x = int(w/2)
    ind_pix_y = int(h/2)

    while(True):
        ok, frame = ip_cam.read()

        
        imgage_colors = np.asarray(frame, dtype='uint8')
        print(imgage_colors[ind_pix_y][ind_pix_x])
        color = imgage_colors[ind_pix_y][ind_pix_x]

        cv2.rectangle(frame, (int(w/2-50), int(h/2-100)), (int(w/2+50), int(h/2+100)), (int(color[0]), int(color[1]), int(color[2])), 5)
        cv2.rectangle(frame, (int(w/2-100), int(h/2-50)), (int(w/2+100), int(h/2+50)), (int(color[0]), int(color[1]), int(color[2])), 5)

        if not ok:
            break
        cv2.imshow("video", frame)

        if(cv2.waitKey(24) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

#rect_on_image()
#write_video_ip_cam()
get_video_and_get_RGB_color()