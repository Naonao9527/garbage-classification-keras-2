import numpy as np
import colorsys
#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

model_path      = 'model_data/best_epoch_weights.h5'
litemodel_path  = "raspi/model.tflite"
classes_path    = 'model_data/垃圾_classes.txt'
anchors_path    = 'yolo_anchors.txt'

nms_iou         = 0.3
confidence      = 0.5
input_shape     = [416, 416]
anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
class_names, num_classes = get_classes(classes_path)
anchors, num_anchors     = get_anchors(anchors_path)

 #---------------------------------------------------#
#   画框设置不同的颜色
#---------------------------------------------------#
hsv_tuples  = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors_ = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors_))
