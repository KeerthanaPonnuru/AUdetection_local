import time
import cv2
import os
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'
import onnxruntime
import numpy as np
import cv2
from PIL import Image
from detector import detect_faces
from model import SwinModel
from align import FaceAlign
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
session = onnxruntime.InferenceSession("export.onnx")
inname = [input.name for input in session.get_inputs()]
outname = [output.name for output in session.get_outputs()]

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
    input = {
        inname[0]: image_data,
    }
    t0 = time.time()
    logits = session.run(outname, input)
    logits= np.array(logits[0]).squeeze()
    logits = 1/(1 + np.exp(-logits))
    predict_time = time.time() - t0
    print("Predict Time: %ss" % (predict_time))
    logits[logits>=0.5] = 1
    logits[logits<0.5] = 0 
    return logits



labels = np.array([1,2,4,5,6,7,9,10,12,14,15,17,20,23,24,25,26])
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
        org = (700, 200+(i+1)*50)
        au = AU_DESCRIPTION['AU'+str(au)]
        frame = cv2.putText(frame, au, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return frame

def igen_frames():
    obj=FaceAlign(244,2.9)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the appropriate codec based on file extension
    video = cv2.VideoWriter('bp4d_ouput.mp4', fourcc, 30, (1040, 1392))
    min_face_size = 20.0,
    thresholds=[0.6, 0.7, 0.8],
    nms_thresholds=[0.7, 0.7, 0.7]
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        bounding_boxes, facial_landmarks = detect_faces(pil_image, min_face_size, thresholds, nms_thresholds)
        if ret==True:
            aligned_img=obj.__call__(pil_image, facial_landmarks) 
            image = frame_process(aligned_img)
            outputs = get_prediction(image)
            frame = write_on_frame(frame,outputs)
            video.write(frame)
        else:
            break
        #pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        #bounding_boxes, facial_landmarks = detect_faces(pil_image, min_face_size, thresholds, nms_thresholds)

        #print(facial_landmarks)
        
        #print(aligned_img)

        #for bbox, landmarks in zip(bounding_boxes, facial_landmarks):
         #   x1, y1, x2, y2, z = bbox.astype(int)

        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        #for i in range(5):
         #   x_landmark, y_landmark = int(landmarks[i]), int(landmarks[i + 5])
          #  cv2.circle(aligned_img, (x_landmark, y_landmark), 2, (0, 0, 255), -1)  
        #cv2.imshow('frame',aligned_img)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        #return aligned_img
igen_frames()
'''cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose the appropriate codec based on file extension
video = cv2.VideoWriter('bp4d_ouput.mp4', fourcc, 30, (1040, 1392))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret ==True:
        image = frame_process(frame)
        outputs = get_prediction(image)
        frame = write_on_frame(frame,outputs)
        video.write(frame)
    else:
        break'''
#cap.release()
cv2.destroyAllWindows()
#video.release()