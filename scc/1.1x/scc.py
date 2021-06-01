#%tensorflow_version 1.x
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import pandas as pd
import math

import tensorflow as tf
from tensorflow.keras.models import load_model

set_th_value=150; radius=10; circularity_parameter=0.77

def scc(read, model, set_th_value=150, radius=10, circularity_parameter=0.77):
  if circularity_parameter > 0.79:
      circularity_parameter = 0.79
  t1 = time.time()
  area_parameter = (3.14*radius**2)/0.95 # do NOT change this formula!
  area_parameter = area_parameter/1.04 # do NOT change this formula!

  img = read
  th_value=set_th_value
  test_thresh, __ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)      
  if test_thresh < np.mean(img):
      img = img
  else:
      img = cv2.bitwise_not(img)    
  new_th, __ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  if area_parameter > 200:
      if th_value/np.mean(img) > 0.63:
          a = 5; b = 3; th_value = th_value; sig = 'first'   
          if th_value > new_th:
              a = 5; b = 3; th_value = new_th
      else:
          a = 3; b = 1                            
  else:    
      if th_value > new_th:
          a = 3; b = 1; th_value = new_th
      else:            
          a = 3; b = 1; sig = 'fifth'
          
  ret, img2 = cv2.threshold(img, th_value, 255, cv2.THRESH_BINARY)
  c_function = cv2.medianBlur(img2, 5)
  c_function = cv2.morphologyEx(c_function, cv2.MORPH_CLOSE, np.ones((a,a)), iterations = b)
  d_function = cv2.erode(c_function, np.ones((1,1)), iterations = 1)
  if len(np.unique(d_function)) == 1:
      counts = 0
      return counts
    
  contours, hierarchy = cv2.findContours(d_function, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

  A = []
  df = []
  k = np.empty((1,2))
  for j in range(0, len(contours)):
      M = cv2.moments(contours[j])
      areas = M['m00']
      A.append(areas)

  smallArea = np.where(np.array(A)<area_parameter)
  contours = np.delete(np.array(contours), np.asarray(smallArea).astype('int'))

  for j in range(0, len(contours)):
      M = cv2.moments(contours[j])
      huMoments = cv2.HuMoments(M)
      huMoments = huMoments[0]
      huMoments = -1*math.copysign(1.0, huMoments)*math.log10(abs(huMoments))
      df.append(huMoments)

  df = np.array(df)
  multCirc = np.where(df<circularity_parameter)[0][1:]
  if len(multCirc) == 0:
    counts = len(df)
  else:
    c = [contours[i] for i in multCirc]
    count_array = []
    imgs = []
    for h,cnt in enumerate(c):
      mask = np.ones((img.shape[0]+100, img.shape[1]+100),np.uint8)
      cv2.drawContours(mask,[cnt],0,0,-1)
      (x,y,w,h) = cv2.boundingRect(cnt)
      x1 = x-100; x2 = x+100; y1 = y-100; y2 = y+100
      if x1<0:
          x1=0
      if y1<0:
          y1=0
      if x2<0:
          x2=0
      if y2<0:
          y2=0
      mask_subset = mask[y1:y2, x1:x2]
      mask_subset_resize = cv2.resize(mask_subset, (100, 100))
      im4 = np.array(mask_subset_resize)
      imgs.append(im4)
    imgs = np.array(imgs).reshape(-1, 100, 100, 1)
    #count_array = model.predict_on_batch(imgs)
    #count_array = [np.argmax(i) for i in count_array]
    count_array = [2]

    counts = len(df)-len(multCirc)-1+int(sum(np.array(count_array)))
    t2 = time.time()
    times = t2-t1
  #return counts
  return counts
