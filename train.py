# Setup detectron2 logger

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import cv2
import random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog

import json
import os
import pandas as pd
import csv
import time
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
#from detectron2.evaluation.coco_evaluation import COCOEvaluator
import os

import datetime 
from config import  TRAFFIC_SIGN_ANNOT_TRAIN_JSON, TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_TRAIN_IMG, PATH_TRAIN_IMG, PATH_VALID_IMG, PATH_MODEL_ZOO, PATH_MODEL_FINAL, PATH_TEST_IMG

# using now() to get current time  
current_time = datetime.datetime.now() 

name_train = 'train_' + str(current_time.year) + '_' +  str(current_time.month) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute)  + '_' + str(current_time.second) 
name_valid = 'valid_' + str(current_time.year) + '_' +  str(current_time.month) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute) + '_' + str(current_time.second)  
register_coco_instances(name_train, {}, TRAFFIC_SIGN_ANNOT_TRAIN_JSON, PATH_TRAIN_IMG)
register_coco_instances(name_valid, {}, TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_TRAIN_IMG)

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()


cfg.merge_from_file(model_zoo.get_config_file(PATH_MODEL_ZOO))
cfg.DATASETS.TRAIN = (name_train,)
cfg.DATASETS.TEST = (name_valid,)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(PATH_MODEL_ZOO)  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 2050 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05


cfg.OUTPUT_DIR = PATH_MODEL_FINAL
cfg.SOLVER.CHECKPOINT_PERIOD = 100



cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 200


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()