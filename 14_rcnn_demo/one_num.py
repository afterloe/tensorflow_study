#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from cv2 import imshow, waitKey, imread
from imutils.paths import list_images

from tensorflow import compat, keras, saved_model, convert_to_tensor, newaxis, cast
from numpy import asarray, int64, uint8

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

utils_ops.tf = compat.v2


def load_model(model_name: str):
    base_url = "http://download.tensorflow.org/models/object_detection/"
    model_file = model_name + ".tar.gz"
    model_dir = keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True
    )

    model_dir = model_dir + "/" + "saved_model"
    model = saved_model.load(str(model_dir))
    return model


PATH_TO_LABELS = r"D:\Projects\models-master\research\object_detection\data\mscoco_label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)

# t = r"D:\Projects\models-master\research\object_detection\test_images"
t = r"D:\Projects\models-master\research\object_detection\test_images"
TEST_IMAGE_PATHS = list_images(t)

model_name = r"ssd_mobilenet_v1_coco_2017_11_17"
detection_model = load_model(model_name)


def run_inference_for_single_image(model, image):
    image = asarray(image)
    input_tensor = convert_to_tensor(image)
    input_tensor = input_tensor[newaxis, ...]
    model_fn = model.signatures["serving_default"]
    output_dict = model_fn(input_tensor)
    num_detections = int(output_dict.pop("num_detections"))
    output_dict = {
        key: value[0, :num_detections].numpy() for key, value in output_dict.items()
    }
    output_dict["detection_classes"] = output_dict["detection_classes"].astype(
        int64)

    if "detection_masks" in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict["detection_masks"], output_dict["detection_boxes"], image.shape[0], image.shape[1])
        detection_masks_reframed = cast(detection_masks_reframed > 0.5, uint8)
        output_dict["detection_masks_reframed"] = detection_masks_reframed.numpy()

    return output_dict


def show_inference(model, image_path):
    image_np = imread(image_path)
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(image_np, output_dict["detection_boxes"], output_dict["detection_classes"], output_dict["detection_scores"], category_index, instance_masks=output_dict.get(
        "detection_masks_reframed", None), use_normalized_coordinates=True, line_thickness=8)

    imshow("view", image_np)
    waitKey(0)


idx: int = 0
for image_path in TEST_IMAGE_PATHS:
    show_inference(detection_model, image_path)
    idx += 1
    if idx > 2:
        break
