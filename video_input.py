import cv2
import numpy as np
import argparse
import sys
import os

sys.path.append('/usr/local/lib/python3.9/site-packages')

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", nargs='?', help="path to video")
parser.add_argument("--weights_path", help="path to yolo weights")
args = parser.parse_args()


def compile_video(name):
    images_path = '/Users/isabellayu/datasets/images/'
    vid_path = '/Users/isabellayu/darknet/data/' + name + '.mp4'
    frames = sorted([i for i in os.listdir(images_path) if name in i])
    print(frames)
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (640, 480))
    os.chdir(images_path)
    for frame in frames:
        img = cv2.imread(frame)
        out.write(img)
    out.release()
    return vid_path


def load_yolo():
    net = cv2.dnn.readNet(args.weights_path,
                          "/Users/isabellayu/darknet/cfg/children.cfg")
    classes = []
    with open("/Users/isabellayu/darknet/data/children.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 6))[0]
    return net, classes, colors, output_layers


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids


def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) + ": " + str(round(confs[i], 3))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)


def start_video(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, colors, class_ids, classes, frame)
        cv2.imshow("vid", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
    start_video(args.video_path)
