from math import pi
import cv2
import numpy as np
import pandas as pd
import os

#################### PARAMETERS ####################

mode = 'count'									#(string)	(count, thresh, debug). What mode SCC will operate in. 'counting' will count the cells in each image, 
												#			'thresh' will quickly return background and threshold values, 'debug' will ouput debug images and csv columns.
imageLocation = r'10X images'						#(string)	Directory in which the images to be processed are located.
csvOutputLocation = r''							#(string)	Directory into which output csv file with cell counts will be placed. If blank (''), 
												#			will write to imageLocation.
imageOutputLocation = r''						#(string)	Directory into which debug outputs will be placed. If blank (''), will write to directory of the input images.
size = 13										#(int)		Approximate size of cells to count.
circularityThresh = 0.82						#(float)	How circular a contour has to be before it is counted.
threshMethod = 'rel'							#(string)	(abs: absolute, rel: relative, dif: difference). See below. 

### ABSOLUTE THRESHOLD - Uses a single threshold value across images.
absProcessThresh = 144							#(int)		The brightness used to create the first black-and-white image.
absCellThresh = 134								#(int)		The brightness below which the average pixel brightness within a contour must be to be counted as a cell.

### RELATIVE THRESHOLD - Determines threshold value for each image as a proportion of background brightness.
relProcessThresh = 0.83							#(float)	The brightness used to create the first black-and-white image.
relCellThresh = 0.78							#(float)	The brightness below which the average pixel brightness within a contour must be to be counted as a cell.

### DIFFERENCE THRESHOLD - Determines threshold for each image as a difference from background brightness.
difProcessThresh = 30							#(int)		The brightness used to create the first black-and-white image.
difCellThresh = 40								#(int)		The brightness below which the average pixel brightness within a contour must be to be counted as a cell.

#################### PROCESSING ####################

#Output parameters to console.
print('\n------------------------\n Mode:',mode, '\n Input Directory:',imageLocation, '\n Cells size:', size, '\n Circularity:',circularityThresh, '\n Threshold method:',threshMethod)
if threshMethod == 'abs':
	print('   Processing Threshold:',absProcessThresh, '\n   Cell Threshold:',absCellThresh)
elif threshMethod == 'rel':
	print('   Processing Threshold:',relProcessThresh, '\n   Cell Threshold:',relCellThresh)
elif threshMethod == 'dif':
	print('   Processing Threshold:',difProcessThresh, '\n   Cell Threshold:',difCellThresh)
else:
	print('Invalid Threshold Method Selected! (Initialization)')
	quit()
print('------------------------\n')

imList = []
counts = []
lengths = []
thr = 0
processThresh = 0
cellThresh = 0
backgrounds = []
processThresholds = []
cellThresholds = []

for root, dirs, files in os.walk(imageLocation, topdown=False):
	#Skip file if it has '_' in it, denoting some sort of output file.
	for f in files:
		if 'abs' in f or 'dif' in f or 'rel' in f or '.csv' in f:
			continue
		
		#Put together file names and paths for input and output images.
		imList.append(f)
		inputImg = os.path.join(root, f)
		print(inputImg, flush=True)
		if imageOutputLocation == '':
			outputImg = root + '/' + f[:-4] + '_' + threshMethod + f[-4:]
		else:
			outputImg = imageOutputLocation + '/' + f[:-4] + '_' + threshMethod + f[-4:]
		
		images = [0,0,0,0,0,0]			#The object used to store the image at each stage of transformation
		contours = [[],[],[],[]]		#The object used to store the list of contours after each stage of filtering
		brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
		
	#Load image and transform it
		images[0] = cv2.imread(inputImg)
		images[1] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
		images[5] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
		
		#Assess background brightness and calculate thresholds
		thr = cv2.threshold(images[1], 0, 255, cv2.THRESH_OTSU)[0]
		if threshMethod == 'abs':
			processThresh = absProcessThresh
			cellThresh = absCellThresh
		elif threshMethod == 'rel':
			processThresh = int(thr*relProcessThresh)
			cellThresh = int(thr*relCellThresh)
		elif threshMethod == 'dif':
			processThresh = thr-difProcessThresh
			cellThresh = thr-difCellThresh
		else:
			print('Invalid Threshold Method Selected! (Process)')
			quit()
		#Record thresholds used
		if mode != 'count':
			backgrounds.append(thr)
			processThresholds.append(processThresh)
			cellThresholds.append(cellThresh)
		if mode == 'thresh':
			continue
		
		images[1] = cv2.threshold(images[1], processThresh, 255, cv2.THRESH_BINARY)[1]
		images[2] = cv2.medianBlur(images[1], 5)
		images[3] = cv2.morphologyEx(images[2], cv2.MORPH_CLOSE, brush)
		
	#Find contours and filter them
		contours[0] = cv2.findContours(images[3], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		
		#Length and corner filter
		for i in range(len(contours[0])):
			length = len(contours[0][i])
			if ((contours[0][i][0][0].all() != 0) & (length > 10) & (length < size*4)):
				contours[1].append(contours[0][i])

		#Circularity filter
		for i in range(len(contours[1])):
			moment = cv2.moments(contours[1][i])
			circularity = (moment['m00'] ** 2)/(2*pi * (moment['mu02'] + moment['mu20']))
			if circularity > circularityThresh:
				contours[2].append(contours[1][i])

		#Cell darkness filter
		intensities = []
		for i in range(len(contours[2])):
			mask = np.zeros_like(images[0])
			cv2.drawContours(mask, contours[2], i, color=255, thickness=-1)
			# Access the image pixels and create a 1D numpy array then add to list
			pts = np.where(mask == 255)[:2]
			intensities.append(int(np.mean(images[5][pts[0], pts[1]])))
			if intensities[i] < cellThresh:
				contours[3].append(contours[2][i])

#################### OUTPUT ####################
		
		counts.append(len(contours[3]))
		
	#Debug image generation
		if mode == 'debug':
		#Draw contours
			images[4] = cv2.drawContours(images[0], contours[0], -1, (0,0,255), 2)
			images[4] = cv2.drawContours(images[0], contours[1], -1, (0,140,255), 2)
			images[4] = cv2.drawContours(images[0], contours[2], -1, (0,255,255), 2)
			images[4] = cv2.drawContours(images[0], contours[3], -1, (0,220,0), 2)
			
		#Notate counts
			cv2.putText(images[4], '+!Length:'+str(len(contours[0])), (8,144), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
			cv2.putText(images[4], '+!Circular:'+str(len(contours[1])), (8,108), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,140,255), 2)
			cv2.putText(images[4], '+!Dark:'+str(len(contours[2])), (8,72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
			cv2.putText(images[4], 'Cells:'+str(len(contours[3])), (8,36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,220,0), 2)
			
		#Write debug image
			cv2.imwrite(outputImg, images[4])

#csv generation
		#Record average contour length
			lenList = []
			for i in range(len(contours[3])):
				lenList.append(len(contours[3][i]))
			if lenList == []:
				lengths.append('N/A')
			else:
				lengths.append(np.mean(lenList))

#Structure csv file and add average values to last row
if mode == 'debug':
	save_output = pd.DataFrame({'image':imList,'count':counts,'background':backgrounds,'processThreshold':processThresholds,'cellThreshold':cellThresholds,'length':lengths})
elif mode == 'thresh':
	save_output = pd.DataFrame({'image':imList,'background':backgrounds,'processThreshold':processThresholds,'cellThreshold':cellThresholds})
else:
	save_output = pd.DataFrame({'image':imList,'count':counts})
save_output.set_index('image', inplace=True)
save_output.loc['mean'] = save_output.mean(skipna=True)

#Save csv file
if csvOutputLocation == '':
	csvOutputLocation = imageLocation
save_output.to_csv(csvOutputLocation + '/.' + mode + '_' + threshMethod + '.csv')

print('All Done!')