#visualize training data

import random
from detectron2.utils.visualizer import Visualizer
from config import  TRAFFIC_SIGN_ANNOT_TRAIN_JSON,  PATH_TRAIN_IMG, NAME_TRAIN \
TRAFFIC_SIGN_ANNOT_VALID_JSON, PATH_VALID_IMG, NAME_VALID


def visualize(mode, No_img)
  if mode == 1 :
    name_visualization = NAME_TRAIN
  else:
    name_visualization = NAME_VALID

  my_dataset_metadata = MetadataCatalog.get(name_visualization)
  dataset_dicts = DatasetCatalog.get(name_visualization)
  for d in random.sample(dataset_dicts, No_img):
      img = cv2.imread(d["file_name"])
      visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_metadata, scale=0.5)
      vis = visualizer.draw_dataset_dict(d)
      cv2_imshow(vis.get_image()[:, :, ::-1])
  
def main(args):
  visualize(args['mode'], args['No_img'])

if __name__ == "__main__" :
  parser = argparse.ArgumentParser(description='Process a croped image and produce GOOD or NOT GOOD result.')
  parser.add_argument('--mode', '-m', required=True, 
                      type=bool, help='If you want visualize train data  mode = 1, otherwise mode = 2')
  parser.add_argument('--No_img', '-n', required=True, 
                      type=bool, help='Number of image will visualize')
  args = vars(parser.parse_args())
  main(args)