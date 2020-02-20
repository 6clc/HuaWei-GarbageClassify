## 介绍

1. 使用FastAI fine-tune DenseNet161模型，最终准确略为93.85%，初赛排名50。
2. [赛事链接](https://competition.huaweicloud.com/information/1000007620/introduction)
3. **团队名称:** chaoliu6c
4. [数据整理代码](https://github.com/6clc/DataRearrangement)

## 测试
1. 下载[训练参数](none)到路径dir
2. 把dir填入inferencer/custom_service.py code 3 的 model_dir
3. 使用方式如下
```bash
cd inferencer
python
from custom_service import garbage_predict
garbage_predict('/home/hanshan/Data/DataCV/DataSets/huawei-garbage-classification/00-其他垃圾-一次性快餐盒/img_1.jpg')
```
 