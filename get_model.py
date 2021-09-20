from model import Deeplabv3
from tensorflow.python.keras.metrics import Metric
import keras.backend as K
import numpy as np
import tensorflow as tf

def get_mean_iou(nclasses):
    def mean_iou(y_true, y_pred):

        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.compat.v1.metrics.mean_iou(y_true, y_pred_, nclasses)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)
    return mean_iou

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice
def iou_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou

def model_define(img_w, img_h, classes_in , model_name, load_weigths='pascal'):
    """
    Define model to use:
    img_w,img_h,classes_in(int): Parameters for input tensor.
    model_name(str): Model name to use.
    load_weigths(str): If model have pre-trained weights, set name.
    return model
    """
    if model_name =='Deeplabv3_xception':
        model = Deeplabv3(input_shape=(img_w, img_h, 3),classes=classes_in, backbone='xception')
    if model_name =='Deeplabv3_Mobilenet':
        model = Deeplabv3(input_shape=(img_w, img_h, 3),classes=classes_in, backbone='mobilenetv2')
    return model