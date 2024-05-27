import cv2
import torch

import onnxruntime as ort
import numpy as np
from common import *


ort_sess = ort.InferenceSession('weights.onnx')

lst = []
with open('eval.txt', 'r') as f:
    for filename in f.readlines():
        lst.append(filename.rstrip())

classes = ['Регистрация (постановка на учет)', 'Снятие с учета']
colors = [(0, 255, 0), (0, 0, 255)]

for filename in lst:
    print(filename)

    im = cv2.imread('images/' + filename)
    
    img = letterbox(im, new_shape=(416, 320), auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    
    
    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    outputs, _, _ = ort_sess.run(None, {'input': img.numpy()})
    
    outputs = torch.from_numpy(outputs)
    
    output = non_max_suppression(outputs,
                                conf_thresh=0.5,
                                iou_thresh=0.2,
                                multi_label=False)
    
    
    for detect_index, detect_result in enumerate(output):
        # Assign values based on detection mode
        if not detect_result is None and len(detect_result):
            # Rescale boxes from image_size to raw_frame size
            detect_result[:, :4] = scale_coords(img.shape[2:], detect_result[:, :4], im.shape).round()
    
            # Write results
            for *xyxy, confidence, classes in reversed(detect_result):
                xmin = int(xyxy[0])
                ymin = int(xyxy[1])
                xmax = int(xyxy[2])
                ymax = int(xyxy[3])

                cv2.rectangle(im, (xmin, ymin), (xmax,ymax), colors[int(classes)], 3)

    cv2.imwrite('results/' + filename, im)