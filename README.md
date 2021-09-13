# Robodk-CNN-for-segmentation
This repository presents a dataset created through the use of robodk software, all this for the segmentation of tools and elements in a workstation for a robotic arm.

# EXECUÇÃO CODIGOS POC2

## Obtendo as referências

As informações foram organizadas no formato .shp, que contém informações sobre os polígonos de anotações de cada uma das imagens disponíveis. Para isto efeito fillpolly_GTM.py irá extrair as máscaras usando a data disponíveis nas imagens de entrada.

```bash
python fillpolly_GTM.py  --id_gpu 0 --main_path "/default_path_folders/" --save_path "./CSV_RLE/"
```

## Parâmetros:

* `main_path`(str): Caminho onde se encontra os folders com arquivos '.tiff'.
* `save_path`(str): Caminho onde o arquivo .CSV com a lista de RLE será salvo.


## Treinando um modelo

No processo de treinamento, será utilizado o .csv extraído na etapa anterior. Os endereços das imagens ao lado do RLE serão usados ​​para extrair os patches com as classes de interesse. Os modelos disponíveis serão, Unet, Unet ++, Unet ++ inception (Inception), Unet ++ Densenet201 (Densenet201), Unet ++ inceptionresnetv2 (inceptionresnetv2), Deeplabv3_xception, Deeplabv3_Mobilenet.

```bash
python run Train_Seg.py --id_gpu 0 --size 512 --stride 0.11 --classes 2 --csv_path "./CSV_RLE/" --model_name "Inception" --model_weigths_dir "./Results/Models/09-21/IC/011_CLAHE/001_170EPOCHStile64.h5" --epochs 200 --Flip_x True --Flip_y True --zoom True --CLAHE True --batch 8
```

## Parâmetros:

* `size`(int): Tamanho do patch e da entrada do modelo.
* `classes`(int): classes de interesse.
* `model_name`(str): modelo a ser usado, Unet, Unet ++, Inception,Densenet201, inceptionresnetv2, Deeplabv3_xception e Deeplabv3_Mobilenet.
* `model_weigths_dir`(str): diretório para salvar os pesos treinados.
* `epochs`(int): Epocas de treino. 
* `Flip_x`(bool): uso do flip no axis x.
* `Flip_y`(bool): uso do flip no axis y.  
* `zoom`(bool): uso do zoom.  
* `CLAHE`(bool): uso do CLAHE. 
* `batch`(int): batch_size. 


## Executando o processo de inferência

Para o processo de inferência, todas as imagens foram utilizadas, fazendo-se uso de inferência em lote para o total de patches por imagem. Como resultado, as imagens correspondentes serão geradas para a imagem de entrada, máscara, inferência e área de inferência, juntamente com o relatório de métricas em formato .CSV

python run inference.py --id_gpu 0 --size 512 --classes 2 --model_name "Inception" --model_weigths_dir "./Results/Models/09-21/IC/011_CLAHE/001_170EPOCHStile64.h5" --out_path_inf '/Results/Inferences/XC_32TILE' --save_plot True 

## Parâmetros:
* `size`(int): Tamanho do patch e da entrada do modelo.
* `classes`(int): classes de interesse.
* `model_name`(str): modelo a ser usado, Unet, Unet ++, Inception,Densenet201, inceptionresnetv2, Deeplabv3_xception e Deeplabv3_Mobilenet.
* `model_weigths_dir`(str): diretório para salvar os pesos treinados.
* `out_path_inf`(str): diretório para salvar as inferências.
* `save_plot True`(bool): Flag para salvar os plots.

### Avaliando os resultados

Para a avaliação dos resultados, incluiu-se o uso das métricas RECALL, PRECISION e F1-score. Esta seção se concentrará no estudo do desempenho geral de algumas métricas com base na ordenação por valor da métrica.

```bash
python get_global_metric_curve.py --id_gpu 0 --inference_path '/Results/Inferences/XC_32TILE' --path_to_save '/Results/Inferences/XC_32TILE' --metric 'recall_feicao'
```

### Parâmetros:
* `metric `(str): metrica de interesse, recall_feicao e f1_feicao.
* `path_to_save`(str): diretório para salvar a curva.
* `inference_path`(str): diretório das inferências.


