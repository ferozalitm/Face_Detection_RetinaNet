# Author: T M Feroz Ali
# Date: 14 AMy 2025
#
# Versions
# 1. RetinaFace-tf2/detect_dataset.py
#           Take path to a dataset and outputs BBox overlayed image (one img at a time).

import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from src.retinafacetf2.retinaface import RetinaFace

from argparse import ArgumentParser
import os
import glob
from tqdm import tqdm


flags.DEFINE_float('det_thresh', 0.9, "detection threshold")
flags.DEFINE_float('nms_thresh', 0.4, "nms threshold")
flags.DEFINE_bool('use_gpu_nms', True, "whether to use gpu for nms")

parser = ArgumentParser()
parser.add_argument('--dataset_src_path', type=str, default = '../dataset_examples/EgoSurf/Original_resolution', help = "dataset_src_path")
parser.add_argument('--dataset_destination_path', type=str, default = './Results/EgoSurf', help = "dataset_destination_path")
args = parser.parse_args()




def _main(_argv):
    detector = RetinaFace(FLAGS.use_gpu_nms, FLAGS.nms_thresh)

    # Open the dataset foolder and create list of all the images to be detected.
    source_folder = args.dataset_src_path
    dest_folder = args.dataset_destination_path
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # breakpoint()

    img_file_list_jpg = sorted(glob.glob(source_folder + "/**/*.jpg", recursive=True))    
    img_file_list_png = sorted(glob.glob(source_folder + "/**/*.png", recursive=True))  
    img_file_list = []
    if len(img_file_list_jpg) != 0:
        img_file_list.extend(img_file_list_jpg) 
    if len(img_file_list_png) != 0:
        img_file_list.extend(img_file_list_png) 

    num = 0

    # breakpoint()
    for img_file in tqdm(img_file_list):
        img = cv2.imread(img_file)
        # breakpoint()
        img_relative_path = img_file.split(source_folder)[1].strip('/')
        dest_img_path = os.path.join(dest_folder, img_relative_path)
        # Use .jpg in export image
        dest_img_path = dest_img_path if dest_img_path.endswith('.jpg') else dest_img_path.split('.png')[0] + '.jpg'


        img = cv2.imread(img_file)
        faces, landmarks = detector.detect(img, FLAGS.det_thresh)

        if faces is not None:
            print('found', faces.shape[0], 'faces')
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                color = (0, 0, 255)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    for l in range(landmark5.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 1)

        # breakpoint()
        if not os.path.exists(os.path.dirname(dest_img_path)):
            os.makedirs(os.path.dirname(dest_img_path))
        cv2.imwrite(dest_img_path, img)
        num +=1

    print('Num of images processed and saved ', num)


if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
