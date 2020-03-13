import cv2
import argparse

def get_images_from_video(video_name, time_F, output):

    vc = cv2.VideoCapture(video_name)
    c = 1
    
    if vc.isOpened(): #判斷是否開啟影片
        rval, video_frame = vc.read()
    else:
        rval = False

    while rval:   #擷取視頻至結束
        rval, video_frame = vc.read()
        
        if(c % time_F == 0): #每隔幾幀進行擷取
            
            cv2.imwrite(output + video_name.split('/')[-1].split('.')[0] + '_' +str(c)+ '.jpg', video_frame)
            print(video_name.split('/')[-1].split('.')[0] + '_f' +str(c)+ '.jpg')
            
        c = c + 1
    vc.release()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action='store', dest='input')
    parser.add_argument("--fps", type=int, dest='fps')
    parser.add_argument("--output", action='store', dest='output', default='./output_frame/')
    
    args = parser.parse_args()
    time_F = args.fps #time_F越小，取樣張數越多
    video_name = args.input #影片名稱
    get_images_from_video(video_name, time_F, args.output) #讀取影片並轉成圖片