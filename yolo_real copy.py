import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt

#YOLO MODELİ ÇAĞIRMA
model=cv2.dnn.readNetFromONNX('static/models/best.onnx')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

INPUT_WIDTH = 640
INPUT_HEIGHT= 640


def yolo_real_time(path,filename):
    cap=cv2.VideoCapture(path)
    while True:
        ret,frame=cap.read()

        if ret==False:
            print('unable to read video')
            break
        results=yolo_predictions(frame,model)

        cv2.namedWindow("YOLO",cv2.WINDOW_KEEPRATIO)
        cv2.imshow("YOLO",results)
        if cv2.waitKey(1)==27: #ESC
            break
    cv2.destroyAllWindows()
    cap.release()
    



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
    index=cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45).flatten()

    return boxes_np,confidences_np,index
def drawings(image,boxes_np,confidences_np,index):
    #drawings
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
        return ''
    else:
        text=pt.image_to_string(roi)
        text=text.strip()

        return text


           