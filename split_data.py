import glob
from sklearn.model_selection import train_test_split
import shutil
import os

# PATH 

images_path = "Data/fake"


train_img_dir = "data_set/train/fake"

test_img_dir = "data_set/test/fake"

val_img_dir = "data_set/val/fake"

# lay name file

filenames = list()
files_images = glob.glob(os.path.join(images_path, '*.jpg'))
for fil in files_images:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    filenames.append(filename)
print(len(filenames))

# chia name file random
filename_train_val, filename_test = train_test_split(filenames, test_size=0.2, shuffle=True)
filename_train, filename_val = train_test_split(filename_train_val, test_size=0.1, shuffle=True)

# # # chia train val test
print("train:", len(filename_train))
print("test:", len(filename_test))
print("val:", len(filename_val))

for train in filename_train:
    image_path = os.path.join(images_path, f"{train}.jpg")
    shutil.move(image_path, train_img_dir)
print("done")
for val in filename_val:
    image_path = os.path.join(images_path, f"{val}.jpg")
    shutil.move(image_path, val_img_dir)
print("done")
for test in filename_test:
    image_path = os.path.join(images_path, f"{test}.jpg")
    shutil.move(image_path, test_img_dir)
print("done")