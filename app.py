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
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
factor=1.5
app = Flask(__name__)
CORS(app) 
session = onnxruntime.InferenceSession("swin_transformer.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]
DYNAMIC_TEXT=''
VAL=[]

def frame_process(frame, input_shape=(224, 224)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, input_shape)
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    return image


def get_prediction(image_data):
    val=[]
    input = {
        inname[0]: image_data,
    }
    t0 = time.time()
    logits = session.run(outname, input)
    logits= np.array(logits[0]).squeeze()
    logits = 1/(1 + np.exp(-logits))
    predict_time = time.time() - t0
    val=[float(value) for value in logits]
    val=[x * 100 for x in val]
    return val



labels = np.array([1,2,4,6,7,9,10,12,14,15,17,20,23,24,25,26,27,43])
AU_DESCRIPTION = {
                    'AU1' : 'Inner Brow Raiser',
                    'AU2' : 'Outer Brow Raiser',
                    'AU4' : 'Brow Lowerer',
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


def fun():
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(-0)
    obj=FaceAlign(224,2.9)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    videos = cv2.VideoWriter('ouput.mp4', fourcc, 5, (1280, 960))
    min_face_size = 20.0
    thresholds=[0.6, 0.7, 0.8]
    nms_thresholds=[0.7, 0.7, 0.7]
    x1,y1,x2,y2,z=0,0,0,0,0
    bounding_boxes=[]
    landmarks=[]
    while (cap.isOpened()):
        ret, frame = cap.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)
        mesh_frame = np.full(frame.shape, 255, dtype=np.uint8)
        if result.multi_face_landmarks:

            for face_landmarks in result.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    height, width, _ = frame.shape
                    center_y=height//2
                    center_x=width//2
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    xx=center_x + int((x - center_x) * factor)
                    yy=center_y + int((y - center_y) * factor)
                    cv2.circle(mesh_frame, (xx, yy), 2, (0, 0, 0), -1)

        pil_image = Image.fromarray(frame)
        bounding_boxes, facial_landmarks = detect_faces(pil_image, min_face_size, thresholds, nms_thresholds)
        aligned_img=obj(pil_image, facial_landmarks)
        for bbox, landmarks in zip(bounding_boxes, facial_landmarks):
            x1, y1, x2, y2, z = bbox.astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            

            for i in range(5):
                x_landmark, y_landmark = int(landmarks[i]), int(landmarks[i + 5])
                cv2.circle(frame, (x_landmark, y_landmark), 2, (0, 0, 255), -1) 
            if ret ==True:
                ret, buffer = cv2.imencode('.jpg', frame)
                image = frame_process(aligned_img)
                val = get_prediction(image)
                if not ret:
                    continue

                global VAL 
                VAL=val
                output_frame = np.hstack((frame, mesh_frame))

                ret, buffer = cv2.imencode('.jpg', output_frame)
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(fun(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/percentage')
def percentage():
    global VAL
    response_datas = {
        
        'global_data': VAL
    }
    return jsonify(response_datas)


if __name__ == '__main__':
    app.run(debug=True)

