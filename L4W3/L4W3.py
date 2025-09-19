# YOLO 车辆识别
import argparse
import os

import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    box_scores = box_confidence * box_class_probs

    # 每个区域最高分的索引及分数
    box_classes = tf.argmax(box_scores, axis=-1)
    box_class_scores = tf.reduce_max(box_scores, axis=-1)

    # 生成掩码
    filtering_mask = box_class_scores >= threshold

    # 保留分数较高的信息
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_class_scores, filtering_mask)

    return scores, boxes, classes

# box_confidence = tf.random.normal([19, 19, 5, 1], mean=1.0, stddev=4.0, seed=1)
# boxes = tf.random.normal([19, 19, 5, 4], mean=1.0, stddev=4.0, seed=1)
# box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1.0, stddev=4.0, seed=1)

# scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
#
# print(scores[2])
# print(boxes[2])
# print(classes[2])
# print(scores.shape)
# print(boxes.shape)
# print(classes.shape)

def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

# box1 = (2, 1, 4, 3)
# box2 = (1, 2, 3 ,4)
#
# print(iou(box1, box2))

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    nms_indices = tf.image.non_max_suppression(
        boxes,
        scores,
        max_output_size=max_boxes,
        iou_threshold=iou_threshold
    )

    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                      iou_threshold=iou_threshold)

    return scores, boxes, classes

# yolo_outputs = (
#     tf.random.normal([1, 19, 19, 5, 1], mean=1, stddev=4, seed=1),  # box_confidence
#     tf.random.normal([1, 19, 19, 5, 2], mean=1, stddev=4, seed=1),  # box_xy
#     tf.random.normal([1, 19, 19, 5, 2], mean=1, stddev=4, seed=1),  # box_wh
#     tf.random.normal([1, 19, 19, 5, 80], mean=1, stddev=4, seed=1)  # box_class_probs
# )

# scores, boxes, classes = yolo_eval(yolo_outputs, image_shape=(720., 1280.))
#
# print("scores[2] =", scores[2].numpy())
# print("boxes[2] =", boxes[2].numpy())
# print("classes[2] =", classes[2].numpy())
#
# print("scores.shape =", scores.numpy().shape)
# print("boxes.shape =", boxes.numpy().shape)
# print("classes.shape =", classes.numpy().shape)

def predict(image_file, model, class_names, scores, boxes, classes, model_image_size=(608, 608)):
    # 预处理图像
    image, image_data = preprocess_image("images/" + image_file, model_image_size)
    image_data = tf.convert_to_tensor(image_data, dtype=tf.float32)  # 转为 Tensor

    yolo_outputs = model(image_data)
    out_scores, out_boxes, out_classes = yolo_eval(
        yolo_outputs,
        image_shape=(image.size[1], image.size[0]),
        max_boxes=10,
        score_threshold=0.6,
        iou_threshold=0.5
    )

    # 将张量转换为 NumPy 数组
    out_scores = out_scores.numpy()
    out_boxes = out_boxes.numpy()
    out_classes = out_classes.numpy().astype(int)

    # 打印预测信息
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    # 生成颜色用于绘制边界框
    colors = generate_colors(class_names)

    # 在图像上绘制边界框
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)

    # 保存图像
    os.makedirs("out", exist_ok=True)
    image.save(os.path.join("out", image_file), quality=90)

    # 显示图像
    output_image = plt.imread(os.path.join("out", image_file))
    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()

    return out_scores, out_boxes, out_classes

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720.,1280.)

# tf.config.enable_unsafe_deserialization()
yolo_model = load_model("model_data/yolov2.h5",
                        compile=False,
                        safe_mode=False)

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

out_scores, out_boxes, out_classes = predict('test.jpg', yolo_model, class_names, scores, boxes, classes)