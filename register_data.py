from detectron2.data.datasets import register_coco_instances
import datetime
from config import  TRAFFIC_SIGN_ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN \\
TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID \\
TRAFFIC_SIGN_ANNOT_TEST_JSON, PATH_TEST_IMG, NAME_VALID \\
PATH_MODEL_ZOO,PATH_MODEL_FINAL

current_time = datetime.datetime.now() 
time = str(current_time.year) + '_' +  str(current_time.month) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute)  + '_' + str(current_time.second) 

register_coco_instances(NAME_TRAIN + time, {}, TRAFFIC_SIGN_ANNOT_TRAIN_JSON, PATH_TRAIN_IMG)
register_coco_instances(NAME_VALID + time, {}, TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_TRAIN_IMG)
register_coco_instances(NAME_TEST + time, {}, TRAFFIC_SIGN_ANNOT_TEST_JSON, PATH_TEST_IMG)
