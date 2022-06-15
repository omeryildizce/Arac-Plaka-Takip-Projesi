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


def OCR(path,filename):

    img = cv2.imread(path)
    img=cv2.resize(img,(1600,1200))
    result_img,text = yolo_predictions(img,model,filename)
    return text



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
def drawings(image,boxes_np,confidences_np,index,filename):
    #drawings
    for ind in index:
        x,y,w,h=boxes_np[ind]
        bb_conf=confidences_np[ind]
        conf_text='plate: {:.0f}%'.format(bb_conf*100)
        license_text=extract_text(image,boxes_np[ind])


        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(x+w,y),(255,0,255))
        cv2.rectangle(image,(x,y+h),(x+w,y+h+30),(0,0,0),-1)

        roi = image[y:y+h,x:x+w]
        roi_bgr = cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        cv2.imwrite('static/roi/{}'.format(filename),roi_bgr)


        cv2.putText(image,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(image,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),1)

    return image,license_text

#predictions
def yolo_predictions(img,net,filename):
    ##1: predictions 
    input_image,detections= get_detections(img,net)
    ##2: NMS
    boxes_np,confidences_np,index=non_maximum_suppression(input_image,detections)
    ##3: Drawings
    results_img,text=drawings(img,boxes_np,confidences_np,index,filename)
    cv2.imwrite('static/predict/{}'.format(filename),results_img)
    return results_img,text

def extract_text(image,bbox):
    x,y,w,h=bbox
    roi=image[y-5:y+h+5, x-5:x+w+5]

    if 0 in roi.shape:
        return ''
    else:
        text=pt.image_to_string(roi)
        text=text.strip()

        return text


           