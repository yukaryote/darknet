import albumentations as A
import cv2 as cv
import os

DATA_PATH = '/Users/isabellayu/datasets/images/'

transform = A.Compose([
    A.RandomCrop(width=450, height=450),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo'))

images = []
labels = []

for i in os.listdir(DATA_PATH):
    if i.endswith('.jpg'):
        images.append(i)
    elif i.endswith('.txt'):
        labels.append(i)

images, labels = sorted(images), sorted(labels)

os.chdir(DATA_PATH)

for ix, im in enumerate(images):
    image = cv.imread(im)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    bbox_path = labels[ix]
    bboxes = []
    with open(bbox_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            class_name = line.pop(0)
            for i in range(len(line)):
                line[i] = float(line[i])
            line.append(class_name)
            bboxes.append(line)
        print(bboxes)
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    print(im, ix)
    trans_img_path = DATA_PATH + im[:-4] + "_trans.jpg"
    trans_label_path = DATA_PATH + im[:-4] + "_trans.txt"
    if not os.path.isfile(trans_img_path):
        cv.imwrite(trans_img_path, transformed_image)
    if not os.path.isfile(trans_label_path):
        with open(trans_label_path, "w") as f:
            for line in transformed_bboxes:
                line = list(line)
                class_name = line.pop()
                line.insert(0, class_name)
                f.write(" ".join([str(num) for num in line]) + "\n")