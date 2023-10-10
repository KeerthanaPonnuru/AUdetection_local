import time
import cv2 
import onnxruntime
import numpy as np
from PIL import Image
from flask import Flask, Response, render_template,jsonify
from detector import detect_faces
from align import FaceAlign
import threading
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 
session = onnxruntime.InferenceSession("swin_transformer.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]
DYNAMIC_TEXT=''


def frame_process(frame, input_shape=(224, 224)):
    print('hi')

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


def get_prediction(image_data):

    input = {
        inname[0]: image_data,
    }
    t0 = time.time()
    logits = session.run(outname, input)
    #print(logits)
    logits= np.array(logits[0]).squeeze()
    logits = 1/(1 + np.exp(-logits))
    predict_time = time.time() - t0
    #print("Predict Time: %ss" % (predict_time))
    logits[logits>=0.5] = 1
    logits[logits<0.5] = 0
    return logits



labels = np.array([1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26,27,43])
AU_DESCRIPTION = {
                    'AU1' : 'Inner Brow Raiser',
                    'AU2' : 'Outer Brow Raiser',
                    'AU4' : 'Brow Lowerer',
                    'AU5' : 'Upper Lid Raiser',
                    'AU6' : 'Cheek Raiser',
                    'AU7' : 'Lid Tightener',
                    'AU9' : 'Nose Wrinkler',
                    'AU10' : 'Upper Lip Raiser',
                    'AU12' : 'Lip Corner Puller',
                    'AU14' : 'Dimpler',
                    'AU15' : 'Lip Corner Depressor',
                    'AU17' : 'Chin Raiser',
                    'AU20' : 'Lip Stretcher',
                    'AU23' : 'Lip Tightener',
                    'AU24' : 'Lip Pressor',
                    'AU25' : 'Lips Part',
                    'AU26' : 'Jaw Drop',
                    'AU27' : 'Mouth Stretch',
                    'AU43' : 'Eyes Closed'
                 }

def write_on_frame(frame,outputs):

    aus = labels[np.argwhere(outputs).squeeze()]
    #print(aus)
    window_name = 'Image'
  
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # fontScale
    fontScale = 1
    
    # Blue color in BGR
    color = (0, 0, 255)
    
    # Line thickness of 2 px
    thickness = 2

    for i, au in enumerate(aus):
        org = (400, 100+(i+1)*50)
        au = AU_DESCRIPTION['AU'+str(au)]
        #print(au)
        frame = cv2.putText(frame, au, org, font, fontScale, color, thickness, cv2.LINE_AA)
        #print(frame)
    return frame,au



def fun():
    cap = cv2.VideoCapture(-0)
    obj=FaceAlign(244,2.9)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    videos = cv2.VideoWriter('ouput.mp4', fourcc, 2, (640, 480))
    min_face_size = 20.0
    thresholds=[0.6, 0.7, 0.8]
    nms_thresholds=[0.7, 0.7, 0.7]
    while (cap.isOpened()):
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        bounding_boxes, facial_landmarks = detect_faces(pil_image, min_face_size, thresholds, nms_thresholds)
        
        for bbox, landmarks in zip(bounding_boxes, facial_landmarks):
            x1, y1, x2, y2, z = bbox.astype(int)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        for i in range(5):
            x_landmark, y_landmark = int(landmarks[i]), int(landmarks[i + 5])
            cv2.circle(frame, (x_landmark, y_landmark), 2, (0, 0, 255), -1) 
        if ret ==True:
            ret, buffer = cv2.imencode('.jpg', frame)
            aligned_img=obj(pil_image, facial_landmarks)
            image = frame_process(frame)
            outputs = get_prediction(image)

            frame,dynamic_text = write_on_frame(frame,outputs)
            print(dynamic_text,0)
            dynamic_text_bytes = dynamic_text.encode('utf-8')
            if not ret:
                continue

            DYNAMIC_TEXT=dynamic_text
            #print(DYNAMIC_TEXT,1)
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n' +
                b'Content-Type: text/plain\r\n\r\n' + dynamic_text_bytes + b'\r\n')
            #yield (jsonify({'caption': captions}),)
            #yield (b'--frame\r\n'
               #b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            #yield (b'--frame\r\n'
             #          b'Content-Type: text/plain\r\n\r\n' + dynamic_text.encode() + b'\r\n')
            #return captions
            #yield (b'--frame\r\n'
             #  b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
              # b'X-Text-Caption: ' + dynamic_text.encode() + b'\r\n')

    #cap.release()

#def text():
 #   return ''


@app.route('/')
def index():
    dynamic_text = DYNAMIC_TEXT
    return render_template('index.html',dynamic_text=dynamic_text)

@app.route('/video')
def video():
    return Response(fun(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
