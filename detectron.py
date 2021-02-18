from detectron2.utils.logger import setup_logger
setup_logger()
import os
import cv2 as cv
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

images_path = 'datasets/images/'
labels_path = 'datasets/labels/'
images = os.listdir(images_path)
os.chdir(images_path)

for img in images:
    print(img)
    im = cv.imread(img)
    outputs = predictor(im)
    classes = outputs['instances'].pred_classes
    boxes = outputs["instances"].pred_boxes
    bbox_file = open(labels_path + img[:-4] + ".txt", "w")

    for ix, c in enumerate(classes):
        if c == 0:
            box = boxes[ix].tensor[0]
            bbox_file.write(str(c.item()) + "\n")
            print("box", box)
            for coord in box:
                bbox_file.write(str(coord.item()) + " ")
    bbox_file.close()