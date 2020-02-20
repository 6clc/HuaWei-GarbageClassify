import numpy as np
import torch
from PIL import Image
from fastai.vision import *
from torchvision.models import *
import pretrainedmodels

from fastai.vision import *
from fastai.vision.models import *
from fastai.vision.learner import model_meta
import sys

model_dir = '/home/hanshan/Projects/ProjectsCV/HuaWei-GarbageClassify/inferencer/ckpt/best.pth'
path = '/'
classes = [item for item in range(40)]
data2 = ImageDataBunch.single_from_classes(
    path, classes, size=224, resize_method=ResizeMethod.SQUISH).normalize(imagenet_stats)


def se_resnext50(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return model


class garbage_classify_service():
    def __init__(self):
        self.model_path = model_dir

        # self.learn = create_cnn(data2, se_resnext50, pretrained=False, cut=-2)
        self.learn = cnn_learner(data2, models.densenet201, pretrained=False)
        self.learn.load(self.model_path.replace('.pth', ''))

        self.label_id_name_dict = {
                "0": "其他垃圾/一次性快餐盒",
                "1": "其他垃圾/污损塑料",
                "2": "其他垃圾/烟蒂",
                "3": "其他垃圾/牙签",
                "4": "其他垃圾/破碎花盆及碟碗",
                "5": "其他垃圾/竹筷",
                "6": "厨余垃圾/剩饭剩菜",
                "7": "厨余垃圾/大骨头",
                "8": "厨余垃圾/水果果皮",
                "9": "厨余垃圾/水果果肉",
                "10": "厨余垃圾/茶叶渣",
                "11": "厨余垃圾/菜叶菜根",
                "12": "厨余垃圾/蛋壳",
                "13": "厨余垃圾/鱼骨",
                "14": "可回收物/充电宝",
                "15": "可回收物/包",
                "16": "可回收物/化妆品瓶",
                "17": "可回收物/塑料玩具",
                "18": "可回收物/塑料碗盆",
                "19": "可回收物/塑料衣架",
                "20": "可回收物/快递纸袋",
                "21": "可回收物/插头电线",
                "22": "可回收物/旧衣服",
                "23": "可回收物/易拉罐",
                "24": "可回收物/枕头",
                "25": "可回收物/毛绒玩具",
                "26": "可回收物/洗发水瓶",
                "27": "可回收物/玻璃杯",
                "28": "可回收物/皮鞋",
                "29": "可回收物/砧板",
                "30": "可回收物/纸板箱",
                "31": "可回收物/调料瓶",
                "32": "可回收物/酒瓶",
                "33": "可回收物/金属食品罐",
                "34": "可回收物/锅",
                "35": "可回收物/食用油桶",
                "36": "可回收物/饮料瓶",
                "37": "有害垃圾/干电池",
                "38": "有害垃圾/软膏",
                "39": "有害垃圾/过期药物"
            }


    def _inference(self, img):
        pred_class, pred_idx, outputs = self.learn.predict(img)
        pred_label = int(pred_class)
        result = {'result': self.label_id_name_dict[str(pred_label)]}
        return result


def garbage_predict(img_path):
    img = open_image(img_path)
    tester = garbage_classify_service()
    return tester._inference(img)['result']
