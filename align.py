import numpy as np
import cv2
class FaceAlign():
    def __init__(self,img_size,enlarge = 2.9):
        self.size = img_size
        self._enlarge = enlarge
  
    def __call__(self, img, landmarks):
        if(img):
            
            if len(landmarks)>0:
                landmarks = landmarks[0].reshape((-1,2),order='F')
                dx = landmarks[1,0] - landmarks[0,0]
                dy = landmarks[1,1] - landmarks[0,1]
                l = np.sqrt(dx**2 + dy**2)
                sinVal = dy / l
                cosVal = dx / l
                mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
                mat2 = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
            
                mat2 = (mat1 * mat2.T).T
                cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
                cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
                if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
                    halfSize = 0.5 * self._enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
                else:
                    halfSize = 0.5 * self._enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))
                scale = (self.size - 1) / 2.0 / halfSize
                mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
                mat = mat3 * mat1
                img1=np.array(img)
                aligned_img = cv2.warpAffine(img1, mat[0:2, :], (self.size, self.size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            else:
                aligned_img=[]
                print('No landmarks detected')
            return aligned_img
    