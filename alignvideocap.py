import cv2
from PIL import Image
from detector import detect_faces
from align import FaceAlign

def igen_frames():
    obj=FaceAlign(244,2.9)
    cap = cv2.VideoCapture(0)
    min_face_size = 20.0,
    thresholds=[0.6, 0.7, 0.8],
    nms_thresholds=[0.7, 0.7, 0.7]
    
    while(True):
        ret, frame = cap.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        bounding_boxes, facial_landmarks = detect_faces(pil_image, min_face_size, thresholds, nms_thresholds)
        aligned_img=obj(pil_image, facial_landmarks)

        for bbox, landmarks in zip(bounding_boxes, facial_landmarks):
            x1, y1, x2, y2, z = bbox.astype(int)

        cv2.rectangle(aligned_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        for i in range(5):
            x_landmark, y_landmark = int(landmarks[i]), int(landmarks[i + 5])
            cv2.circle(aligned_img, (x_landmark, y_landmark), 2, (0, 0, 255), -1)  
        cv2.imshow('frame',aligned_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        return aligned_img
igen_frames()
cv2.destroyAllWindows() 