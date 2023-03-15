from email.mime import image
from mtcnn import MTCNN
import os
import glob
import cv2


import cv2
import os

def crop_images_from_folder(folder, path_dir):
    # images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            detector = MTCNN()
            detections = detector.detect_faces(img)
            min_cof = 0.9
            print(filename)
            for det in detections:
                if det['confidence'] >= min_cof:
                    x,y,w,h = det['box']
                    crop_img = img[y:y+h, x:x+w]
                    # obj = cv2.resize(crop_img, dim)
                    path_dir1 = os.path.join(path_dir, f"{filename}")
                    cv2.imwrite(path_dir1,crop_img)
                    print(path_dir1)

if __name__=='__main__':
    # PATH = ["ClientRaw/0001", "ClientRaw/0002", "ClientRaw/0003","ClientRaw/0004","ClientRaw/0005","ClientRaw/0006","ClientRaw/0007","ClientRaw/0008","ClientRaw/0009","ClientRaw/0010","ClientRaw/0011","ClientRaw/0012","ClientRaw/0013","ClientRaw/0014","ClientRaw/0015","ClientRaw/0016"]
    PATH = ["ImposterRaw/0001","ImposterRaw/0002","ImposterRaw/0003","ImposterRaw/0004","ImposterRaw/0005","ImposterRaw/0006","ImposterRaw/0007","ImposterRaw/0008","ImposterRaw/0009","ImposterRaw/0010","ImposterRaw/0011","ImposterRaw/0012","ImposterRaw/0014","ImposterRaw/0015","ImposterRaw/0016"]
    path_dir = "Data/fake"
    for i in PATH:
        crop_images_from_folder(i, path_dir)




















