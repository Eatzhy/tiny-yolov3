# tiny-yolov3
使用tiny——yolov3（keras）检测自己的数据集，三类目标

程序是根据github上yolov3修改的，所以大面积重复，使用tiny-yolo用法如下：

1、下载tiny-yolov3工程，打开yolo.docx文档，按照文档中的教程对自己的
图像集做标注，并生成一些必须的图像路径txt文件。

2、训练图像使用 tiny_train.py
训练后的权重文件会保存在logs下

3、对待测图像进行批量测试：
yolo_test_batch.py
然后会在VOC/SegmentationClass生成检测后的结果

我的程序是在ubantu下跑的，当然改一下路径之类的就可以在windows下测试啦

有问题欢迎讨论，加微信
![微信扫一扫](https://github.com/Eatzhy/tiny-yolov3/blob/master/IMG_5707.JPG)
