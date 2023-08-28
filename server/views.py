from flask import Blueprint,request,current_app,send_file,jsonify
from . import segmentation_model as model
import cv2
from PIL import Image
import numpy as np
import base64,io


global res
res = ''
views = Blueprint('views',__name__)

@views.route('/',methods=['POST','GET'])
def home():
    #executor.submit(bg_predict)
    return 'Hello'

@views.route('/result',methods=['GET'])
def result():
    img = cv2.cvtColor(cv2.imread('1.png'),cv2.COLOR_BGR2RGB)
    pred = model.predict(np.expand_dims(img/255,axis = 0))[0]
    cv2.imwrite('crop.png',img*pred)
    return send_file('../../crop.png',mimetype='image/png')


@views.route('/custom_predict',methods = ['POST'])
def custom_predict():
    file = request.json
    npimg = np.array(file['data']['data'],dtype=np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    if img.shape != (512,512,3):
        img = cv2.resize(img,(512,512))
    cv2.imwrite('input.png',img)
    pred = model.predict(np.expand_dims(img/255,axis=0))[0]
    crop = img*pred
    cv2.imwrite('crop.png',crop)
    with open('crop.png', 'rb') as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8') 
    #if pred != 'Normal':
        #executor.submit(send_mail,pred)
    print('sending...')
    return jsonify({'status':str(base64_image)})

