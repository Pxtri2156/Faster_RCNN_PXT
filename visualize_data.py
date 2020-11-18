#visualize training data
my_dataset_train_metadata = MetadataCatalog.get("my_dataset_train_btc_0903pm_14_11_2020")
dataset_dicts = DatasetCatalog.get("my_dataset_train_btc_0903pm_14_11_2020")

import random
from detectron2.utils.visualizer import Visualizer

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=my_dataset_train_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2_imshow(vis.get_image()[:, :, ::-1])