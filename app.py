from wsgiref.util import request_uri
from flask import Flask, render_template, request
 
import os 
from deeplearning1 import OCR
from deeplearning import OCR_resnet
from deeplearningdif import OCR_dif
from yolo_real import yolo_real_time
from youtube import yolo_real_time_youtube

app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')


@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR(path_save,filename)

        return render_template('index.html',upload=True,upload_image=filename,text=text)

    return render_template('index.html',upload=False)

@app.route('/yolo',methods=['POST','GET'])
def yolo():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        yolo_real_time(path_save,filename)

        return render_template('yolo.html',upload=True,upload_image=filename)

    return render_template('yolo.html',upload=False)

@app.route('/resnet',methods=['POST','GET'])
def resnet():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR_resnet(path_save,filename)

        return render_template('resnet.html',upload=True,upload_image=filename,text=text)

    return render_template('resnet.html',upload=False)

@app.route('/resnet_dif',methods=['POST','GET'])
def resnet_dif():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH,filename)
        upload_file.save(path_save)
        text = OCR_dif(path_save,filename)

        return render_template('resnet_dif.html',upload=True,upload_image=filename,text=text)

    return render_template('resnet_dif.html',upload=False)



if __name__ =="__main__":
    app.run(debug=True)