import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
import numpy as np
import youtube_dl
import pytesseract as pt

#YOLO MODELİ ÇAĞIRMA
model=cv2.dnn.readNetFromONNX('static/models/best.onnx')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

INPUT_WIDTH = 640
INPUT_HEIGHT= 640



#-----------------------------------------------------------------------------------
def yolo_real_time_youtube():
    
    
    if __name__ == '__main__':

        video_url = 'https://www.youtube.com/watch?v=RKIPWCWldzE&t=386s&ab_channel=JUtah'

        ydl_opts = {}

        # create youtube-dl object
        ydl = youtube_dl.YoutubeDL(ydl_opts)

        # set video url, extract video information
        info_dict = ydl.extract_info(video_url, download=False)

        # get video formats available
        formats = info_dict.get('formats',None)

        for f in formats:

            # I want the lowest resolution, so I set resolution as 144p
            if f.get('format_note',None) == '1080p':

                #get the video url
                url = f.get('url',None)

                # open url with opencv
                cap = cv2.VideoCapture(url)
    
                # check if url was opened
                if not cap.isOpened():
                    print('video not opened')
                    exit(-1)

                while True:
                    # read frame
                    ret, frame = cap.read()
                    
                    # check if frame is empty
                    if not ret:
                        break

                    results=yolo_predictions(frame,model)

                    cv2.namedWindow("YOLO",cv2.WINDOW_KEEPRATIO)
                    cv2.imshow("YOLO",results)

                    # display frame
                    #cv2.imshow('frame', frame)

                    if cv2.waitKey(30)&0xFF == ord('q'):
                        break

                # release VideoCapture
                cap.release()

        cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------



def get_detections(img,net):
  #resmi yolo formatına çevir
    image=img.copy()
    row,col,d=image.shape
    max_rc=max(row,col)
    input_image=np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col]=image


    #YOLO MODELİNDEN TAHMİN AL
    blob=cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds=net.forward()
    detections=preds[0]
    return input_image, detections
def non_maximum_suppression(input_image,detections):
    # # GÜVEN VE OLASILIK Skoruna Dayalı FİLTRE TESPİTLERİ
    #kolonlar sırasıyla center_x, center_y, w, h, conf, proba değerlerini tutuyor
    boxes=[]
    confidences=[]

    image_w,image_h=input_image.shape[:2]
    x_factor=image_w/INPUT_WIDTH
    y_factor=image_h/INPUT_HEIGHT
    for i in range(len(detections)):
        row=detections[i]
        confidence=row[4] # tespit edilen plaka güven skoru
        if confidence>0.4:
            class_score=row[5] # plakanın olasılık skoru
            if class_score>0.25:
                cx,cy,w,h=row[0:4]

                left=int((cx-0.5*w)*x_factor)
                top=int((cy-0.5*h)*y_factor)
                width=int(w*x_factor)
                height=int(h*y_factor)
                box=np.array([left,top,width,height])

                confidences.append(confidence)
                boxes.append(box)
    #clean
    boxes_np=np.array(boxes).tolist()
    confidences_np=np.array(confidences).tolist()
    #NMS
    index=np.array(cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)).flatten()

    return boxes_np,confidences_np,index
def drawings(image,boxes_np,confidences_np,index):
    #drawings
    license_text=""
    for ind in index:
        x,y,w,h=boxes_np[ind]
        bb_conf=confidences_np[ind]
        conf_text='plate: {:.0f}%'.format(bb_conf*100)
 
        license_text=extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255),-1)
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)



        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image,license_text

#predictions
def yolo_predictions(img,net):
    # 1: predictions 
    input_image,detections= get_detections(img,net)
    # 2: NMS
    boxes_np,confidences_np,index=non_maximum_suppression(input_image,detections)
    # 3: Drawings
    results_img,text=drawings(img,boxes_np,confidences_np,index)
    return results_img

def extract_text(image,bbox):
    x,y,w,h=bbox
    roi=image[y-5:y+h+5, x-5:x+w+5]

    if 0 in roi.shape:
        return ' '
    else:
        text=pt.image_to_string(roi)
        text=text.strip()

        return text


yolo_real_time_youtube()