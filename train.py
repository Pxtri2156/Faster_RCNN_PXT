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
import argparse
from config import  TRAFFIC_SIGN_ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN, \
TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID, \
TRAFFIC_SIGN_ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_TEST, \
PATH_MODEL_ZOO,PATH_MODEL_FINAL
import register_data



class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

def train(pretrain, no_iter):
  cfg = get_cfg()


  cfg.merge_from_file(model_zoo.get_config_file(PATH_MODEL_ZOO))
  cfg.DATASETS.TRAIN = (NAME_TRAIN,)
  cfg.DATASETS.TEST = (NAME_VALID,)

  cfg.DATALOADER.NUM_WORKERS = 4
  if pretrain == True:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(PATH_MODEL_ZOO)  # Let training initialize from model zoo
  else:
    cfg.MODEL.WEIGHTS = ""
  cfg.SOLVER.IMS_PER_BATCH = 4
  cfg.SOLVER.BASE_LR = 0.001


  cfg.SOLVER.WARMUP_ITERS = 1000
  print("NO ITER ", no_iter)
  cfg.SOLVER.MAX_ITER = no_iter #adjust up if val mAP is still rising, adjust down if overfit
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

def main(args):
  train(args['pretrain'], args['no_iter'])

if __name__ == "__main__" :
  parser = argparse.ArgumentParser(description='Config train Faster RCNN')
  parser.add_argument('--pretrain', '-p', default = False, 
                      type=bool, help='pretrain p = 1 otherwise p = 0 ')
  parser.add_argument('--no_iter', '-n', required=True, 
                      type=int, help='Number of iter ')
  args = vars(parser.parse_args())
  main(args)