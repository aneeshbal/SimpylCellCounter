from math import pi
import os
import cv2
import numpy as np
import pandas as pd
import configparser
import imghdr

#Create 'config.ini' if it doesn't exist in the current directory
if os.path.isfile('config.ini') == False:
	cfgFile = open('config.ini','w')
	print('No config file detected!\nGenerating config file.', flush=True)
	lines = [
		'[GENERAL]\n'
		'mode = debug\n'
		'	#(string) - (count, thresh, debug). What mode SCC will operate in.\n'
		'		#\'count\' will count the cells in each image\n'
		'		#\'thresh\' will quickly return background and threshold values\n'
		'		#\'debug\' will output debug images and csv columns.\n'
		'fluorescent = False\n'
		'	#(bool) - Change to true to process fluorescence images\n'
		'threshMethod = rel\n'
		'	#(string) - (abs: absolute, rel: relative, dif: difference). See sections below.\n'
		'imageLocation = 10X images\n'
		'	#(string) - Directory in which the images to be processed are located. Can be full path or path relative to scc.py\n'
		'csvOutputLocation = \n'
		'	#(string) - May be left blank. Directory into which output csv file with cell counts will be placed. If blank, will write to imageLocation. Can be full path or path relative to scc.py\n'
		'imageOutputLocation = \n'
		'	#(string) - May be left blank. Directory into which debug outputs will be placed. If blank, will write to directory of the input images. Can be full path or path relative to scc.py\n\n'
		'[CELL FEATURES]\n'
		'size = 13\n'
		'	#(int) - Approximate size (radius in pixels) of cells to count.\n'
		'minAreaCoeff = 0.8\n'
		'	#(float) - Coefficient of the formula which determines the minimum area a cell can be. (A = size^2 * minAreaCoeff)\n'
		'maxAreaCoeff = 4.5\n'
		'	#(float) - Coefficient of the formula which determines the maximum area a cell can be. (A = size^2 * maxAreaCoeff)\n'
		'circularityThresh = 0.9\n'
		'	#(float) - How circular a cell\'s contour must be before it is counted.\n\n'
		'#Process Thresholds determine the brightness used to create the first black-and-white image.\n'
		'#Cell Thresholds determine the brightness below which the average pixel brightness within a contour must be to be counted as a cell.\n'
		'[ABSOLUTE THRESHOLD] - (int) Uses a single threshold value across images. Higher values are less restrictive.\n'
		'processThresh = 144\n'
		'cellThresh = 134\n\n'
		'[RELATIVE THRESHOLD] - (float) Determines threshold value for each image as a proportion of background brightness. Higher values are less restrictive.\n'
		'processThresh = 0.83\n'
		'cellThresh = 0.78\n\n'
		'[DIFFERENCE THRESHOLD] - (int) Determines threshold for each image as a difference from background brightness. Higher values are more restrictive.\n'
		'processThresh = 30\n'
		'cellThresh = 40'
	]
	cfgFile.writelines(lines)
	cfgFile.close()
	print('\nConfig generation complete! Please view and modify parameters in \'config.ini\' in directory \'' + os.getcwd() +'\'\n', flush=True)
	quit()

#################### PARAMETERS ####################

config = configparser.ConfigParser()
config.read('config.ini')

#Grab parameters from 'config.ini'
mode = config.get('GENERAL','mode')
fluorescent = config.getboolean('GENERAL','fluorescent')
threshMethod = config.get('GENERAL','threshMethod')
imageLocation = config.get('GENERAL','imageLocation')
csvOutputLocation = config.get('GENERAL','csvOutputLocation')
imageOutputLocation = config.get('GENERAL','imageOutputLocation')

size = config.getint('CELL FEATURES','size')
minAreaCoeff = config.getfloat('CELL FEATURES','minAreaCoeff')
maxAreaCoeff = config.getfloat('CELL FEATURES','maxAreaCoeff')
circularityThresh = config.getfloat('CELL FEATURES','circularityThresh')

absProcessThresh = config.getint('ABSOLUTE THRESHOLD','processThresh')
absCellThresh = config.getint('ABSOLUTE THRESHOLD','cellThresh')

relProcessThresh = config.getfloat('RELATIVE THRESHOLD','processThresh')
relCellThresh = config.getfloat('RELATIVE THRESHOLD','cellThresh')

difProcessThresh = config.getint('DIFFERENCE THRESHOLD','processThresh')
difCellThresh = config.getint('DIFFERENCE THRESHOLD','cellThresh')

#################### PROCESSING ####################

#Output parameters to console
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

#Initialize variables
imList = []
counts = []
areas = []
thr = 0
processThresh = 0
cellThresh = 0
backgrounds = []
processThresholds = []
cellThresholds = []

#Calculate minumum and maximum areas for filtering
minArea = size**2 * minAreaCoeff
maxArea = size**2 * maxAreaCoeff

for root, dirs, files in os.walk(imageLocation, topdown=False):
	#Skip file if it has '_' in it, denoting some sort of output file
	for f in files:
		inputImg = os.path.join(root, f)
		if imghdr.what(os.path.join(root, f)) == 'None':
			continue
		if 'abs' in f or 'dif' in f or 'rel' in f:
			continue
		print(inputImg, flush=True)
		
		#Generate file names with paths for input and output images
		imList.append(f)
		xLen = f.rfind('.')
		if imageOutputLocation == '':
			outputImg = root + '/' + f[:xLen] + '_' + threshMethod + f[xLen:]
		else:
			outputImg = imageOutputLocation + '/' + f[:xLen] + '_' + threshMethod + f[xLen:]
		
		images = [0,0,0,0,0,0]			#The object used to store the image at each stage of transformation
		contours = [[],[],[],[]]		#The object used to store the list of contours after each stage of filtering
		brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
		
	#Load image and transform it
		images[0] = cv2.imread(inputImg)
		images[1] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
		if fluorescent:
			images[1] = cv2.bitwise_not(images[1])
		images[5] = images[1]			#Saved for creating the image masks to determine average pixel darkness within a contour
		
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
		
		#Process image
		images[1] = cv2.threshold(images[1], processThresh, 255, cv2.THRESH_BINARY)[1]
		images[2] = cv2.medianBlur(images[1], 5)
		images[3] = cv2.morphologyEx(images[2], cv2.MORPH_CLOSE, brush)
		
	#Find contours and filter them
		contours[0] = cv2.findContours(images[3], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		
		#Area filter
		for i in range(len(contours[0])):
			area = cv2.contourArea(contours[0][i])
			if ((area > minArea) & (area < maxArea)):
				contours[1].append(contours[0][i])

		#Circularity filter
		for i in range(len(contours[1])):
			moment = cv2.moments(contours[1][i])
			circularity = (moment['m00']**2)/(2*pi * (moment['mu02'] + moment['mu20']))
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
			images[4] = cv2.drawContours(images[0], contours[0], -1, (200,0,255), 2)
			images[4] = cv2.drawContours(images[0], contours[1], -1, (0,140,255), 2)
			images[4] = cv2.drawContours(images[0], contours[2], -1, (0,255,255), 2)
			images[4] = cv2.drawContours(images[0], contours[3], -1, (200,220,0), 2)
			
		#Notate counts
			cv2.putText(images[4], '+!Area:'+str(len(contours[0])), (8,144), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,255), 2)
			cv2.putText(images[4], '+!Circular:'+str(len(contours[1])), (8,108), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,140,255), 2)
			cv2.putText(images[4], '+!Dark:'+str(len(contours[2])), (8,72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
			cv2.putText(images[4], 'Cells:'+str(len(contours[3])), (8,36), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,220,0), 2)
			
		#Write debug image
			cv2.imwrite(outputImg, images[4])

#csv generation
		#Record average contour area
			areaList = []
			for i in range(len(contours[3])):
				areaList.append(cv2.contourArea(contours[3][i]))
			if areaList == []:
				areas.append('N/A')
			else:
				areas.append(np.mean(areaList))

#Structure csv file and add average values to last row
if mode == 'debug':
	csv = pd.DataFrame({'image':imList,'count':counts,'background':backgrounds,'processThreshold':processThresholds,'cellThreshold':cellThresholds,'average area':areas})
elif mode == 'thresh':
	csv = pd.DataFrame({'image':imList,'background':backgrounds,'processThreshold':processThresholds,'cellThreshold':cellThresholds})
else:
	csv = pd.DataFrame({'image':imList,'count':counts})
csv.set_index('image', inplace=True)
outputcsv = csv.copy()
outputcsv.loc['mean'] = csv.mean(skipna=True)
outputcsv.loc['sem'] = csv.sem(skipna=True)

#Save csv file
if csvOutputLocation == '':
	csvOutputLocation = imageLocation
outputcsv.to_csv(csvOutputLocation + '/.' + mode + '_' + threshMethod + '.csv')

print('All Done!')
