# Robodk-CNN-for-segmentation
This repository presents a dataset created through the use of robodk software, all this for the segmentation of tools and elements in a workstation for a robotic arm.

# Train code

```bash
Python train_code.py --id_gpu 0 --size 512 --stride 0.11 --classes 12 --model_name "Deeplabv3_xception" --model_weigths_dir  "/scratch/parceirosbr/manntisict/radar/TEST_MODELS/models/__tst.h5" --epochs 1 --batch 2
```

## Parâmetros:

* `size`(int): Input size.
* `classes`(int): number of classe .
* `model_name`(str): Deeplabv3_xception and Deeplabv3_Mobilenet.
* `model_weigths_dir`(str): weigths path .h5.
* `epochs`(int): Train Epochs. 
* `batch`(int): Batch size. 

## Parâmetros:

![alt text](https://github.com/williamalbert94/Robodk-CNN-for-segmentation/tree/main/Example/pngegg.png)

