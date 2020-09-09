from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt
#import seaborn as sns

from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max



class LoadingModel:

	def iou_coef(y_true, y_pred, smooth=1):
	  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
	  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
	  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
	  return iou

	def dice_loss(y_true, y_pred):
	  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
	  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
	  return 1 - (numerator + 1) / (denominator + 1)

	def dice_coef(y_true, y_pred):
	  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
	  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
	  return (numerator + 1) / (denominator + 1)

	def combine_loss(y_true, y_pred):
		def dice_loss(y_true, y_pred):
			numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
			denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))
			return tf.reshape(1 - numerator / denominator, (-1, 1, 1))
		return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

		
	def load_the_model(url):
		model = load_model(url,
					   custom_objects={'combine_loss': LoadingModel.combine_loss,
									   'iou_coef': LoadingModel.iou_coef})
		return model
		
class ProcessMasks:

	def create_mask(image, model):
	    img = image.copy()
	    img = cv2.resize(img, (256, 256))
	    img = np.reshape(img, (1, 256, 256, 3))
		prediction = model.predict(img)
		return prediction
		
	def process_masks(image):
		
		distance = ndi.distance_transform_edt(image)
		local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((15, 15)),
									labels=image)
		markers = ndi.label(local_maxi)[0]
		labels = watershed(-distance, markers, mask=image)
		
		centersx = []
		centersy = []
		for i in range(1, markers.max()):
		  where = np.where(markers == i)
		  centersx.append(np.mean(where[0]))
		  centersy.append(np.mean(where[1]))
		  
		image_to_paste = image.copy()
		for i,j in zip(centersx,centersy):
		  cv2.circle(image_to_paste, (int(j), int(i)), 5, (255), 1)
		  
		return len(markers), image_to_paste
		
		
		
		
		
		
		
		
	
