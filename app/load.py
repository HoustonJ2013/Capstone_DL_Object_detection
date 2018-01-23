import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
from pspnet import *

def init(): 
	# json_file = open('model.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# #load woeights into new model
	# loaded_model.load_weights("model.h5")
	# print("Loaded Model from disk")

	Capnet = PSPNet50(nb_classes=150, input_shape=(473, 473),
					  weights="pspnet50_ade20k")

	#compile and evaluate loaded model
	Capnet.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	graph = tf.get_default_graph()

	return Capnet,graph