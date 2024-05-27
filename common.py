import torch
from torch import Tensor
import numpy as np
import torchvision.ops
import cv2

def box_iou(box1: Tensor or np.ndarray, box2: Tensor or np.ndarray) -> Tensor or np.ndarray:
    r"""Calculate the intersection-over-union (IoU) of boxes.

    Args:
        box1 (Tensor[N, 4]): Tensor containing N boxes in (x1, y1, x2, y2) format.
        box2 (Tensor[M, 4]): Tensor containing M boxes in (x1, y1, x2, y2) format.

    Returns:
        iou (Tensor[N, M]): Tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    def box_area(box):
        """
        Calculate the area of a box.

        Args:
            box (Tensor[4, n]): Tensor containing the coordinates of n boxes in (x1, y1, x2, y2) format.

        Returns:
            area (Tensor[n]): Tensor containing the area of each box.
        """
        return (box[2] - box[0]) * (box[3] - box[1])

    # Calculate the areas of box1 and box2
    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # Calculate the intersection of box1 and box2
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

    # Calculate the IoU
    iou = inter / (area1[:, None] + area2 - inter)

    return iou

def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    r"""Convert bounding boxes from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        x (np.ndarray): Bounding boxes, shaped [N, 4].

    Returns:
        np.ndarray: Converted bounding boxes, shaped [N, 4].
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction: Tensor,
        conf_thresh: float = 0.1,
        iou_thresh: float = 0.6,
        multi_label: bool = True,
        filter_classes: list = None,
        agnostic: bool = False,
) -> Tensor:
    """
    Performs  Non-Maximum Suppression on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """

    # Settings
    # merge for best mAP
    merge = True
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096

    # number of classes
    num_classes = prediction[0].shape[1] - 5
    # multiple labels per box
    multi_label &= num_classes > 1
    output = [None] * prediction.shape[0]
    # Process each image in the prediction
    for img_idx, x in enumerate(prediction):
        # Apply confidence and width-height constraints
        x = x[x[:, 4] > conf_thresh]  # Confidence threshold
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # Width-height constraints

        # If no detections remain, process next image
        if not x.shape[0]:
            continue

        # Compute confidence
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Convert box coordinates from (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Apply multi-label or best class filtering
        if multi_label:
            i, j = (x[:, 5:] > conf_thresh).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # Best class only
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thresh]

        # Filter by class if specified
        if filter_classes:
            x = x[(j.view(-1, 1) == torch.tensor(filter_classes, device=j.device)).any(1)]

        # If no detections remain, process next image
        num_boxes = x.shape[0]  # Number of boxes
        if not num_boxes:
            continue

        # Apply NMS (Non-Maximum Suppression)
        classes = x[:, 5] * 0 if agnostic else x[:, 5]
        boxes, scores = x[:, :4].clone() + classes.view(-1, 1) * max_wh, x[:, 4]  # Adjusted boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thresh)

        # Merge NMS (boxes merged using weighted mean)
        if merge and (1 < num_boxes < 3E3):
            try:
                iou = box_iou(boxes[i], boxes) > iou_thresh  # IoU matrix
                weights = iou * scores[None]  # Box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # Merged boxes
            except:
                print(x, i, x.shape, i.shape)
                pass

        output[img_idx] = x[i]  # Store the selected detections in the output list

    return output

def letterbox(
        img: np.ndarray,
        new_shape: tuple = (416, 416),
        color: tuple = (114, 114, 114),
        auto: bool = True,
        scale_fill: bool = False,
        scaleup: bool = True
) -> tuple:
    """Resize image to a 32-pixel-multiple rectangle.

    Args:
        img (ndarray): Image to resize
        new_shape (int or tuple): Desired output shape of the image
        color (tuple): Color of the border
        auto (bool): Whether to choose the smaller dimension as the new shape
        scale_fill (bool): Whether to stretch the image to fill the new shape
        scaleup (bool): Whether to scale up the image if the image is smaller than the new shape

    Returns:
        ndarray: Resized image

    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes: Tensor, image_shape: tuple) -> Tensor:
    r"""Clip bounding xyxy bounding boxes to image shape (height, width)

    Args:
        boxes (Tensor): xyxy bounding boxes, shape (n, 4)
        image_shape (tuple): (height, width)

    Returns:
        Tensor: Clipped bounding boxes
    """
    boxes[:, 0].clamp_(0, image_shape[1])  # x1
    boxes[:, 1].clamp_(0, image_shape[0])  # y1
    boxes[:, 2].clamp_(0, image_shape[1])  # x2
    boxes[:, 3].clamp_(0, image_shape[0])  # y2
    return boxes

def scale_coords(new_image_shape, coords, raw_image_shape, ratio_pad=None) -> np.ndarray:
    r"""Rescale coordinates (xyxy) from img1_shape to img0_shape.

    Args:
        new_image_shape (tuple): Shape of the new image (height, width).
        coords (np.ndarray): Coordinates to be scaled, shaped [N, 4].
        raw_image_shape (tuple): Shape of the original image (height, width).
        ratio_pad (tuple, optional): Ratio and padding values for rescaling. Defaults to None.

    Returns:
        np.ndarray: Scaled coordinates, shaped [N, 4].
    """
    if ratio_pad is None:  # calculate from img0_shape
        # gain = old / new
        gain = max(new_image_shape) / max(raw_image_shape)
        # wh padding
        pad = (
            (new_image_shape[1] - raw_image_shape[1] * gain) / 2,
            (new_image_shape[0] - raw_image_shape[0] * gain) / 2
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, raw_image_shape)
    return coords