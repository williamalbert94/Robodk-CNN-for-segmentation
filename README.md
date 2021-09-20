# Robodk-CNN-for-segmentation
This repository presents a dataset created through the use of robodk software, all this for the segmentation of tools and elements in a workstation for a robotic arm.

# Train code

```bash
Python train_code.py --id_gpu 0 --size 512 --stride 0.11 --classes 12 --model_name "Deeplabv3_xception" --model_weigths_dir  "/scratch/parceirosbr/manntisict/radar/TEST_MODELS/models/__tst.h5" --epochs 1 --batch 2
```

## Parâmetros:

* `size`(int): Tamanho do patch e da entrada do modelo.
* `classes`(int): classes de interesse.
* `model_name`(str): modelo a ser usado, Unet, Unet ++, Inception,Densenet201, inceptionresnetv2, Deeplabv3_xception e Deeplabv3_Mobilenet.
* `model_weigths_dir`(str): diretório para salvar os pesos treinados.
* `epochs`(int): Epocas de treino. 
* `batch`(int): batch_size. 

