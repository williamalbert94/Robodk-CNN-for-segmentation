from glob import glob
import pandas as pd
import os
from os.path import join, exists, dirname
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras import Model
from sklearn.metrics import classification_report
from get_model import *
from utils_inf import *
from generator import *
import pickle


def run_train(args,train_generator,validation_generator):
	"""
	Start the train process:
	args(argparse): Parameters of argparse input.
	train_gen,val_gen(object): generators.
	"""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		model = model_define(args.size,args.size,args.classes,args.model_name,None)
		opt = adam(lr=0.0001,clipnorm=1.)
		#if args.model_name=='Unet':
		#	from tensorflow.keras.optimizers import Adam
		#	opt = Adam(lr=0.0001,clipnorm=1.)
		model.compile(optimizer =opt , loss='binary_crossentropy',metrics=['accuracy',dice_coef,iou_coef,get_mean_iou(2)])
		steps_per_epoch = train_generator.__len__() // args.batch
		print("total steps_per_epoch :{}".format(steps_per_epoch))

		earlyStopping = EarlyStopping(monitor='val_mean_iou', mode = 'max', patience=16, verbose=1)
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode = 'min', factor=0.0001, patience=6,min_lr=0.0000000001, verbose=1)
		model_checkpoint = ModelCheckpoint(args.model_weigths_dir, monitor='val_mean_iou',
											mode='max', save_best_only=True, verbose=1, period=1)
		validation_steps = validation_generator.__len__() // args.batch
		history = model.fit_generator(train_generator,
										steps_per_epoch=steps_per_epoch,
										epochs=args.epochs,
										verbose=1,
										validation_data=validation_generator,
										validation_steps=validation_steps,
										callbacks=[earlyStopping,reduce_lr,model_checkpoint],
										shuffle=True)

		with open(args.model_weigths_dir.replace('.h5',''), 'wb') as file_pi:
			pickle.dump(history.history, file_pi)
		acc, val_acc, loss, val_loss,iou,val_iou= history.history['acc'],history.history['val_acc'],history.history['loss'], history.history['val_loss'],history.history['mean_iou'], history.history['val_mean_iou']		
		plt.rcParams['axes.facecolor']='white'
		f, axarr = plt.subplots(1 , 3)

		f.set_figwidth(20)
		f.set_figheight(7)
		# Accuracy
		axarr[0].plot(acc)
		axarr[0].plot(val_acc)
		axarr[0].set_title('model accuracy')
		axarr[0].set_ylabel('accuracy')
		axarr[0].set_xlabel('epoch')
		axarr[0].legend(['train', 'valid'], loc='upper left')

		# Loss
		axarr[1].plot(loss)
		axarr[1].plot(val_loss)
		axarr[1].set_title('model loss')
		axarr[1].set_ylabel('loss')
		axarr[1].set_xlabel('epoch')
		axarr[1].legend(['train', 'valid'], loc='upper left')

		axarr[2].plot(iou)
		axarr[2].plot(val_iou)
		axarr[2].set_title('model mean iou')
		axarr[2].set_ylabel('meaniou')
		axarr[2].set_xlabel('epoch')
		axarr[1].legend(['train', 'valid'], loc='upper left')
		plt.savefig(args.model_weigths_dir.replace('.h5','.png'),bbox_inches='tight')



def main(args):
	print('GPU:{}'.format(args.id_gpu))
	os.environ["CUDA_VISIBLE_DEVICES"]=args.id_gpu
	from glob import glob
	from random import shuffle
	list_images = np.asarray(glob('/scratch/parceirosbr/manntisict/radar/TEST_MODELS/Dataset_pedro/Images/*.png'))
	index_dataset = list(range(len(list_images)))
	train_size = int(len(index_dataset)*0.60)
	val_size = int(len(index_dataset)*0.80)
	#shuffle(index_dataset)
	df_train = list(list_images[:train_size])
	df_val = list(list_images[train_size:val_size])
	df_test = list(list_images[val_size:])
	print(len(df_train))

	train_generator = DataGenerator(list_IDs = df_train , batch_size=args.batch,dim=(512,512), n_channels=3,
	                n_classes=12,norm='min_max', transformations='ALL', shuffle=True)

	validation_generator = DataGenerator(list_IDs = df_val , batch_size=args.batch, transformations='ALL',dim=(512,512), n_channels=3,norm='min_max',
	                n_classes=12)

	run_train(args,train_generator,validation_generator)

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description='Inference: Radar QLK images')
      parser.add_argument('--id_gpu', type=str, dest='id_gpu', help='id da gpu', metavar='id_gpu')
      parser.add_argument('--size', type=int, dest='size', help='Input size', metavar='size')
      parser.add_argument('--stride', type=float, dest='stride', help='Input stride', metavar='stride')
      parser.add_argument('--classes', type=int, dest='classes', help='Input classes', metavar='classes')
      parser.add_argument('--model_name', type=str, dest='model_name', help='Input model_name', metavar='model_name')
      parser.add_argument('--model_weigths_dir', type=str, dest='model_weigths_dir', help='Input model_weigths_dir', metavar='model_weigths_dir')
      parser.add_argument('--batch', type=int, dest='batch', help='batch', metavar='batch', default=8)
      parser.add_argument('--epochs', type=int, dest='epochs', help='epochs', metavar='epochs', default=25)
      parser.add_argument('--save_plot', type=bool, dest='save_plot', help='save_plot', metavar='save_plot', default= False)
      args = parser.parse_args()
      main(args)
