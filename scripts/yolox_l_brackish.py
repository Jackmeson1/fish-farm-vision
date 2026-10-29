"""YOLOX-L exp for the video-disjoint Brackish split (640 input).

Same budget as the other detectors: 50 epochs from COCO weights, last 5 without mosaic/mixup.
"""
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.num_classes = 6
        self.data_dir = "/root/fish/data/brk/yolox"  # annotations/{train,valid,test}.json + image dirs
        self.train_ann = "train.json"
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.max_epoch = 50
        self.no_aug_epochs = 5
        self.warmup_epochs = 3
        self.eval_interval = 10
        self.data_num_workers = 12
        self.test_conf = 0.001
        self.nmsthre = 0.65
        self.eval_split = os.environ.get("YOLOX_EVAL_SPLIT", "valid")
        self.val_ann = self.eval_split + ".json"

    def get_dataset(self, cache=False, cache_type="ram"):
        from yolox.data import COCODataset, TrainTransform
        return COCODataset(
            data_dir=self.data_dir, json_file=self.train_ann, name="train", img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache, cache_type=cache_type)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        legacy = kwargs.get("legacy", False)
        return COCODataset(data_dir=self.data_dir, json_file=self.val_ann, name=self.eval_split,
                           img_size=self.test_size, preproc=ValTransform(legacy=legacy))
