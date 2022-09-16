from math import pi
from imghdr import what
import os
from copy import deepcopy
from sys import platform
import re
import cv2
import numpy as np
import pandas as pd
from configparser import ConfigParser
from matplotlib import use
use("TKAgg")
from matplotlib import pyplot as plt
import PySimpleGUI as sg
import tkinter as tk
import threading
from warnings import filterwarnings
filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#SCC LOOP ------------------------
def scc(window,threadNr,saveFile,imFiles,mode,threshOnly,writeImgs,fluorescent,size,minAreaCoeff,maxAreaCoeff,circularityThresh,threshMethod,absImageThresh,absCellThresh,relImageThresh,relCellThresh,difImageThresh,difCellThresh,channel):
    window.write_event_value('THREAD_CPRINT', (str(mode)+' '+str(threadNr)+' running...', 'black on yellow'))
    
    counts = []
    backgrounds = []
    imageThresholds = []
    cellThresholds = []
    areas = []
    csv = []
    concsv = []
    
    #Min/Max Areas, Brush
    minArea = size**2 * minAreaCoeff
    maxArea = size**2 * maxAreaCoeff
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    
    #Make list of images that will be processed
    if channel and mode == 'Batch':
        imgsToProcess = [imFiles[i] for i, e in enumerate(imFiles) if channel in os.path.basename(e)]
    else:
        imgsToProcess = imFiles
    
    #Iterate over images
    for baseImage in imgsToProcess:
        images = [0,0,0,0,0,0]			#The object used to store the image at each stage of transformation
        contours = [[],[],[],[]]		#The object used to store the list of contours after each stage of filtering
        
        #Transform image
        images[0] = cv2.imread(baseImage)
        images[5] = deepcopy(images[0])
        images[1] = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        if fluorescent:
            images[1] = cv2.bitwise_not(images[1])
        
        #Assess background brightness and calculate thresholds
        thr = cv2.threshold(images[1], 0, 255, cv2.THRESH_OTSU)[0]
        if threshMethod == 'Absolute':
            imageThresh = absImageThresh
            cellThresh = absCellThresh
        elif threshMethod == 'Relative' and not fluorescent:
            imageThresh = int(thr*relImageThresh)
            cellThresh = int(thr*relCellThresh)
        elif threshMethod == 'Relative' and fluorescent:
            invThr = 255 - thr
            imageThresh = thr - int((1-relImageThresh)*invThr)
            cellThresh = thr - int((1-relCellThresh)*invThr)
        elif threshMethod == 'Difference':
            imageThresh = thr-difImageThresh
            cellThresh = thr-difCellThresh
        
        if imageThresh < 5:
            imageThresh = 5
        if cellThresh < 5:
            cellThresh = 5
        
        #Record thresholds used
        if mode == 'Batch':
            backgrounds.append(thr)
            imageThresholds.append(imageThresh)
            cellThresholds.append(cellThresh)
            if threshOnly:
                counts.append('N/A')
                areas.append('N/A')
                continue
        
        #Process image
        for t in range(-40,21,4):
            if cellThresh+t < 0 or cellThresh+t > 255:
                continue
            images[2] = cv2.threshold(images[1], cellThresh+t, 255, cv2.THRESH_BINARY)[1]
            images[3] = cv2.medianBlur(images[2], 5)
            images[3] = cv2.morphologyEx(images[3], cv2.MORPH_OPEN, brush)
            images[4] = cv2.morphologyEx(images[3], cv2.MORPH_CLOSE, brush)
            
            #Find contours
            #newCon = cv2.findContours(images[4], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            #for c in contours[0]:
            #    if cv2.moments(c)['m00'] == 0:
            #        continue
            #    center = tuple([int(cv2.moments(c)[m]/cv2.moments(c)['m00']) for m in ['m10','m01']])
            #    for c2 in range(200):
            #        if c2 >= len(newCon):
            #            break
            #        while True:
            #            if cv2.pointPolygonTest(newCon[c2], center, False) == 1:
            #                del newCon[c2]
            #            else:
            #                break
            #            if c2 >= len(newCon):
            #                break
            contours[0].extend(cv2.findContours(images[4], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0])

        #Area filter
        contours[1] = [e for e in contours[0] if cv2.contourArea(e) > minArea and cv2.contourArea(e) < maxArea]
        for c in range(len(contours[1])):
            if c >= len(contours[1]):
                break
            if cv2.moments(contours[1][c])['m00'] == 0:
                continue
            center = tuple([int(cv2.moments(contours[1][c])[m]/cv2.moments(contours[1][c])['m00']) for m in ['m10','m01']])
            for c2 in range(c+1,len(contours[1])):
                if c2 >= len(contours[1]):
                    break
                while True:
                    if cv2.pointPolygonTest(contours[1][c2], center, False) == 1:
                        del contours[1][c2]
                    else:
                        break
                    if c2 >= len(contours[1]):
                        break

        #Cell darkness filter
        intensities = []
        for i in range(len(contours[1])):
            mask = np.zeros_like(images[0])
            cv2.drawContours(mask, contours[1], i, color=255, thickness=-1)
            # Access the image pixels and create a 1D numpy array then add to list
            pts = np.where(mask == 255)[:2]
            intensities.append(int(np.mean(images[1][pts[0], pts[1]])))
        contours[2] = [e for i, e in enumerate(contours[1]) if intensities[i] < cellThresh]

        #Circularity filter
        contours[3] = [e for e in contours[2] if (cv2.moments(e)['m00']**2)/(2*pi * (cv2.moments(e)['mu02'] + cv2.moments(e)['mu20'])) > circularityThresh]

        if mode == 'Batch':
            counts.append(len(contours[3]))
            areas.append(np.mean([cv2.contourArea(e) for e in contours[3]]))
            
        #Image Generation
        if mode == 'Preview' or writeImgs:
            #Draw contours
            #cv2.drawContours(images[5], contours[0], -1, (200,0,255), 2)
            cv2.drawContours(images[5], contours[1], -1, (0,140,255), 2)
            cv2.drawContours(images[5], contours[2], -1, (0,255,255), 2)
            cv2.drawContours(images[5], contours[3], -1, (200,220,0), 2)

            #Notate counts
            cv2.putText(images[5], '+!Area:'+str(len(contours[0])), (8,36), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,255), 2)
            cv2.putText(images[5], '+!Dark:'+str(len(contours[1])), (8,72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,140,255), 2)
            cv2.putText(images[5], '+!Circular:'+str(len(contours[2])), (8,108), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(images[5], 'Cells:'+str(len(contours[3])), (8,144), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,220,0), 2)
        
        if mode == 'Batch':
            if writeImgs:
                outputImg = '_$$$.'.join(baseImage.rsplit('.',1))
                cv2.imwrite(outputImg, images[5])
            if channel:
                concsv.append(np.array(contours[3]))
        
    if mode == 'Batch':
        #csv Generation
        csv = pd.DataFrame({'image':imgsToProcess,'count':counts,'background':backgrounds,'imageThreshold':imageThresholds,'cellThreshold':cellThresholds,'averageArea':areas})
        csv = csv[['image', 'count', 'background', 'imageThreshold', 'cellThreshold', 'averageArea']]

        csv.set_index('image', inplace=True)
        countcsv = csv.copy()
        countcsv.loc['mean'] = csv.mean(skipna=True)
        countcsv.loc['sem'] = csv.sem(skipna=True)
        #Save csv file
        try:
            countcsv.to_csv(saveFile)
        except PermissionError:
            window.write_event_value('THREAD_CPRINT', (str(mode)+' '+str(threadNr)+' failed!', 'black on red'))
            window.write_event_value('CSV_OPEN', 0)
            return 0

        #npz file 
        if channel and not threshOnly:
            contourArray = np.array(concsv, dtype='object')
            np.savez(saveFile[:-4]+'.npz', names=imgsToProcess, contours=contourArray)
        
    elif mode == 'Preview':
        window.write_event_value('PREVIEW_RETURN', images)

    window.write_event_value('THREAD_CPRINT', (str(mode)+' '+str(threadNr)+' finished!', 'black on palegreen'))

def coex(window,threadNr,saveFile,conFiles,coexImgs):
    window.write_event_value('THREAD_CPRINT', ('Coex '+str(threadNr)+' running...', 'black on yellow'))
    
    channelNames = []
    imageNames = []
    imNamesFull = []
    cellContours = []
    areas = []

    for channel in conFiles:
        arrays = np.load(channel, allow_pickle=True)
        imageNames.append(arrays['names'])
        imNamesFull.extend(arrays['names'])
        cellContours.append(arrays['contours'])
        arrays.close()

    coCounts = []
    coContours = []
    contourCenters = [[[tuple([int(cv2.moments(con)[m]/cv2.moments(con)['m00']) for m in ['m10','m01']]) for con in image] for image in channel] for channel in cellContours]

    #new coex analysis attempt 7/20/22
    for refChannel in range(len(cellContours)):
        for image in range(len(cellContours[refChannel])):
            coCount = 0
            imgCoContours = []
            for center in range(len(contourCenters[refChannel][image])):
                intersects = [True if 1 in [cv2.pointPolygonTest(contour, contourCenters[refChannel][image][center], False) for contour in cellContours[compChannel][image]] else False for compChannel in range(len(cellContours)) if compChannel != refChannel]
                if False not in intersects:
                    coCount += 1
                    imgCoContours.append(cellContours[refChannel][image][center])
            coCounts.append(coCount)
            coContours.append(imgCoContours)
            if imgCoContours == []:
                areas.append('N/A')
            else:
                areas.append(np.mean([cv2.contourArea(c) for c in imgCoContours]))

    #csv Generation
    csv = pd.DataFrame({'image':imNamesFull,'count':coCounts,'averageArea':areas})
    csv = csv[['image', 'count', 'averageArea']]

    csv.set_index('image', inplace=True)
    countcsv = csv.copy()
    countcsv.loc['mean'] = csv.mean(skipna=True)
    countcsv.loc['sem'] = csv.sem(skipna=True)
    #Save csv file
    try:
        countcsv.to_csv(saveFile)
    except PermissionError:
        window.write_event_value('THREAD_CPRINT', ('Coex '+str(threadNr)+' failed!', 'black on red'))
        window.write_event_value('CSV_OPEN', 0)
        return 0

    if coexImgs:
        image = 0
        for file, imgCoContours in zip(imNamesFull, coContours):
            if os.path.isfile(file.rsplit('.',1)[0] + '_$$$.' + file.rsplit('.',1)[1]):
                image = cv2.imread(file.rsplit('.',1)[0] + '_$$$.' + file.rsplit('.',1)[1])
            else:
                image = cv2.imread(file)
            cv2.drawContours(image, imgCoContours, -1, (255,255,255), 2)
            cv2.putText(image, 'Colabeled Cells:'+str(len(imgCoContours)), (8,180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imwrite(file.rsplit('.',1)[0] + '_$$%.' + file.rsplit('.',1)[1], image)
    
    window.write_event_value('THREAD_CPRINT', ('Coex '+str(threadNr)+' finished!', 'black on palegreen'))

##############################################################################################################################################################
#GUI Code

#TOOLTIP LIST
def tt(elementID):
    if(elementID) == 'Batch Mode': return 'Runs SCC on all images in selected folder.'
    elif(elementID) == 'Preview Mode': return 'Runs SCC on one image and diplays processing steps.'
    elif(elementID) == 'Thresholds only': return 'Batch mode will not count cells, will output\nthe background brightness of each image.'
    elif(elementID) == 'Make output images': return 'Batch mode will output\nimages with contours drawn.'
    elif(elementID) == 'Thresh Method': return 'Absolute: Uses same brightness on all images.\nRelative: Multiplies background brightness of each image.\nDifference: Subtracts from background brighness of each image.'
    elif(elementID) == 'AbsThr': return '[0,255] Lower values more selective.'
    elif(elementID) == 'RelThr': return '[0,1] Lower values more selective.'
    elif(elementID) == 'DifThr': return '[0,255] Higher values more selective.'
    elif(elementID) == 'Fluorescent': return 'Enable if analyzing fluorescent images.'
    elif(elementID) == 'Size': return 'Approximate radius, in pixels,\nof smallest cells'
    elif(elementID) == 'Min Area Coefficient': return 'Minimum area of an outlined cell\nto pass the size filter.'
    elif(elementID) == 'Max Area Coefficient': return 'Maximum area of an outlined cell\nto pass the size filter.'
    elif(elementID) == 'Circularity Thresh': return '[0,1) How circular an outlined cell must be\nto pass the circularity filter.'
    elif(elementID) == 'Channel Identifier': return '(Optional) Text identifier of\nchannel type in image names.'
    elif(elementID) == 'CoexImgs': return 'Coexpression contours will\nbe drawn on output images.'
    else: return 'NO MATCHING CASE IN TOOLTIP LIST'

#FILE/FOLDER BROWSE FUNCTIONS ---------------------
def folderbrowse():
    folder_name = tk.filedialog.askdirectory(initialdir=None, parent=None)
    return(folder_name)

def filebrowse(filetypes):
    file_name = tk.filedialog.askopenfilename(filetypes=filetypes, initialdir=None, parent=None)
    return(file_name)

def filesbrowse(filetypes):
    file_names = tk.filedialog.askopenfilenames(filetypes=filetypes, initialdir=None, parent=None)
    return(file_names)

#CONFIG IMPORT/EXPORT FUNCTIONS -----------------------
def importcfg(file):
    config.read(file)
    for k in list(values.keys())[5:18]:
        v = config.get('GENERAL', k)
        window[k].update(bool(v) if (v==True or v==False) else v)
    return config.get('GENERAL', 'thr_meth')

def exportcfg(file,values_dict):
    for k in list(values_dict.keys())[5:18]:
        config.set('GENERAL', k, str(values_dict[k]))
    with open(file, 'w') as configfile:
        config.write(configfile)

#Element Update Functions
def thresh_update(thrMeth):
    window['ACOL'].update(thrMeth.startswith('A'))
    window['RCOL'].update(thrMeth.startswith('R'))
    window['DCOL'].update(thrMeth.startswith('D'))

######################## MAIN CODE ########################

config = ConfigParser()
config.add_section('GENERAL')

fig = plt.figure(figsize=(32, 24))
root = sg.tk.Tk()
root.tk.call('tk', 'scaling', fig.dpi/72)
root.destroy()
plt.close()

#Default Variables on Program Start
conFiles = ''
threadNr = 0

mode = 'Batch'
channel = ''
threshOnly = False
writeImgs = False
fluorescent = True
size = 13
minAreaCoeff = 0.6
maxAreaCoeff = 12.0
circularityThresh = 0.8
threshMethod = 'Relative'
absImageThresh = 144
absCellThresh = 134
relImageThresh = 0.8
relCellThresh = 0.7
difImageThresh = 30
difCellThresh = 40

#WINDOW LAYOUT DEFINITIONS
sg.theme('Neutral Blue')

def make_baseWindow():
    first_col = [
        [sg.Menu([['&Config', ['&Import', '&Export']], ['Settings', ['Advanced Settings']]], key='TEST')],

        [sg.B('Batch Mode', button_color=('#000000','#99FF99'), tooltip=tt('Batch Mode'), key='BATCH_MODE'),
         sg.B('Preview Mode', button_color=('#000000','#999999'), tooltip=tt('Preview Mode'), key='PREVIEW_MODE')],
        [sg.In(size=(27,1), key='IMG_FOLDER'),
         sg.In(size=(27,1), visible=False, key='IMG_FILE')],
        [sg.B('Browse Folders', key='IMG_FOLD_BROWSE'),
         sg.B('Browse Images', visible=False, key='IMG_FILE_BROWSE')],
        [sg.Checkbox('Thresholds only', default=False, enable_events=True, tooltip=tt('Thresholds only'), key='THRESH_ONLY')],
        [sg.Checkbox('Make output images', default=False, tooltip=tt('Make output images'), key='WRITE_IMGS')],
        [sg.T('_'*27)],

        [sg.T('Thresh Method', tooltip=tt('Thresh Method')),
         sg.Drop(values=['Absolute','Relative','Difference'], default_value=threshMethod, size=(10,1), enable_events=True, key='THR_METH')],
        [sg.Col([[sg.T('Image Thresh      Cell Thresh', tooltip=tt('AbsThr'))],
                 [sg.Spin([i for i in range(0,256)], initial_value=absImageThresh, size=(6,1), key='ABS_IMG_THR'),
                  sg.T('       '),
                  sg.Spin([i for i in range(0,256)], initial_value=absCellThresh, size=(6,1), key='ABS_CELL_THR')]], pad=(0,0), visible=threshMethod.startswith('A'), key='ACOL'),
         sg.Col([[sg.T('Image Thresh      Cell Thresh', tooltip=tt('RelThr'), key='REL_MODE')],
                 [sg.Spin([i/100 for i in range(0,101)], initial_value=relImageThresh, size=(6,1), key='REL_IMG_THR'),
                  sg.T('       '),
                  sg.Spin([i/100 for i in range(0,101)], initial_value=relCellThresh, size=(6,1), key='REL_CELL_THR')]], pad=(0,0), visible=threshMethod.startswith('R'), key='RCOL'),
         sg.Col([[sg.T('Image Thresh      Cell Thresh', tooltip=tt('DifThr'), key='DIF_MODE')],
                 [sg.Spin([i for i in range(0,256)], initial_value=difImageThresh, size=(6,1), key='DIF_IMG_THR'),
                  sg.T('       '),
                  sg.Spin([i for i in range(0,256)], initial_value=difCellThresh, size=(6,1), key='DIF_CELL_THR')]], pad=(0,0), visible=threshMethod.startswith('D'), key='DCOL')],
        [sg.T('', size=(None,1))]]

    second_col = [
        [sg.Checkbox('Fluorescent', default=fluorescent, tooltip=tt('Fluorescent'), key='FLUO')],
        [sg.T('Size', tooltip=tt('Size')),
         sg.Spin([i for i in range(1,100)], initial_value=size, size=(6,1), key='SIZE')],
        [sg.T('Min Area Coefficient', tooltip=tt('Min Area Coefficient')),
         sg.Spin([i/100 for i in range(0,4000)], initial_value=minAreaCoeff, size=(6,1), key='MIN_A_COEFF')],
        [sg.T('Max Area Coefficient', tooltip=tt('Max Area Coefficient')),
         sg.Spin([i/100 for i in range(0,5000)], initial_value=maxAreaCoeff, size=(6,1), key='MAX_A_COEFF')],
        [sg.T('Circularity Thresh', tooltip=tt('Circularity Thresh')),
         sg.Spin([i/100 for i in range(0,101)], initial_value=circularityThresh, size=(6,1), key='CIRCLE_THR')],
        [sg.T('_'*33)],
        [sg.T('Channel Identifier', tooltip=tt('Channel Identifier')),
         sg.In(size=(17,1), key='CH_ID')],
        [sg.B('Run', button_color=('#000000','#00BBBB'), key='RUN_SCC'),
         sg.Multiline(size=(25,4), autoscroll=True, write_only=True, reroute_cprint=True, disabled=True, key='REPORT')]]

    third_col = [
        [sg.B('Select Contour Files', key='CON_FILES_BROWSE')],
        [sg.Multiline(visible=False, key='CON_FILES'),
         sg.Multiline(size=(55,5), disabled=True, key='CON_FILES_DISPLAY')],
        [sg.Checkbox('Add contours to output images', default=False, tooltip=tt('CoexImgs'), key='COEX_IMGS')],
        [sg.B('Run', button_color=('#000000','#00BBBB'), key='RUN_COEX')]]

    tab1 = [[sg.Column(first_col), sg.VSeparator(), sg.Column(second_col)]]
    tab2 = [[sg.Column(third_col)]]

    layout = [[sg.TabGroup([[sg.Tab('General', tab1), sg.Tab('Coexpression', tab2)]])]]

    return sg.Window('SimpylCellCounter', layout, scaling=1.5, resizable=True, finalize=True)

#WINDOW INITIALIZATION AND RUNNING
baseWindow = make_baseWindow()

while True:
    window, event, values = sg.read_all_windows()

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    elif event == 'Import':
        cfgFile = filebrowse(filetypes=(('Config files', '*.ini'),))
        if cfgFile:
            thresh_update(importcfg(cfgFile))
            sg.cprint('Config Imported!')

    elif event == 'Export':
        cfgFile = tk.filedialog.asksaveasfilename(filetypes = (('Config files', '*.ini'),), defaultextension = (('Config files', '*.ini'),))
        if cfgFile:
            exportcfg(cfgFile, {k : values[k] for k in values.keys()})
            sg.cprint('Config Exported!')

    elif event == 'BATCH_MODE':
        mode = 'Batch'
        window['BATCH_MODE'].update(button_color='#99FF99')
        window['PREVIEW_MODE'].update(button_color='#999999')
        window['IMG_FOLDER'].update(visible=True)
        window['IMG_FILE'].update(visible=False)
        window['IMG_FOLD_BROWSE'].update(visible=True)
        window['IMG_FILE_BROWSE'].update(visible=False)
        window['THRESH_ONLY'].update(disabled=False)
        if not values['THRESH_ONLY']:
            window['WRITE_IMGS'].update(disabled=False)

    elif event == 'PREVIEW_MODE':
        mode = 'Preview'
        window['BATCH_MODE'].update(button_color='#999999')
        window['PREVIEW_MODE'].update(button_color='#99FF99')
        window['IMG_FOLDER'].update(visible=False)
        window['IMG_FILE'].update(visible=True)
        window['IMG_FOLD_BROWSE'].update(visible=False)
        window['IMG_FILE_BROWSE'].update(visible=True)
        window['THRESH_ONLY'].update(disabled=True)
        window['WRITE_IMGS'].update(disabled=True)
    
    elif event == 'IMG_FOLD_BROWSE':
        window['IMG_FOLDER'].update(folderbrowse())

    elif event == 'IMG_FILE_BROWSE':
        window['IMG_FILE'].update(filebrowse(filetypes = (('Image Files', '*.jpg *jpeg *.png *.tif *.tiff'),)))

    elif event == 'THR_METH':
        thresh_update(values['THR_METH'])

    elif event == 'THRESH_ONLY':
        if values['THRESH_ONLY']:
            window['WRITE_IMGS'].update(disabled=True)
        elif not values['THRESH_ONLY']:
            window['WRITE_IMGS'].update(disabled=False)

    elif event == 'RUN_SCC':
        saveFile = ''
        if mode == 'Preview':
            if os.path.isfile(values['IMG_FILE']) and what(values['IMG_FILE']) != 'None':
                imFiles = [values['IMG_FILE']]
                imNames = [os.path.basename(values['IMG_FILE'])]
            else:
                sg.popup('Invalid File!\nPlease Select an Image File.', location=tuple(map(lambda i, j: i + j, window.CurrentLocation(), (200,160))))
                continue
        elif mode == 'Batch':
            imFiles = []
            imNames = []
            if os.path.isdir(values['IMG_FOLDER']):
                for root, dirs, files in os.walk(values['IMG_FOLDER'], topdown=False):
                    imFiles.extend([os.path.join(root, f) for f in files if what(os.path.join(root, f)) != None and '_$$' not in f])
                    imNames.extend([f for f in files if what(os.path.join(root, f)) != None and '_$$' not in f])
            else:
                sg.popup('Invalid Folder!\nPlease Select a Folder.', location=tuple(map(lambda i, j: i + j, window.CurrentLocation(), (200,160))))
                continue
            saveFile = tk.filedialog.asksaveasfilename(filetypes = (('csv files', '*.csv'),), defaultextension = (('csv files', '*.csv'),))

        if saveFile or mode == 'Preview':
            threadNr += 1
            threading.Thread(target=scc, args=(window,
                                               threadNr,
                                               saveFile,
                                               imFiles,
                                               mode,
                                               values['THRESH_ONLY'],
                                               values['WRITE_IMGS'],
                                               values['FLUO'],
                                               int(values['SIZE']),
                                               float(values['MIN_A_COEFF']),
                                               float(values['MAX_A_COEFF']),
                                               float(values['CIRCLE_THR']),
                                               values['THR_METH'],
                                               int(values['ABS_IMG_THR']),
                                               int(values['ABS_CELL_THR']),
                                               float(values['REL_IMG_THR']),
                                               float(values['REL_CELL_THR']),
                                               int(values['DIF_IMG_THR']),
                                               int(values['DIF_CELL_THR']),
                                               values['CH_ID']),
                             daemon=True).start()

    elif event == 'PREVIEW_RETURN':
        fig = plt.figure(figsize=(32, 24))
        labels = ['Original','Original, grayscale','After image threshold','After gap filling','After size filtering','Final']
        for i, im in enumerate(labels):
            fig.add_subplot(2, 3, i+1)
            plt.imshow(cv2.cvtColor(values['PREVIEW_RETURN'][i], cv2.COLOR_RGB2BGR), cmap='gray')
            plt.axis('off')
            plt.title(im)
        plt.tight_layout()
        plt.show()

    elif event == 'CON_FILES_BROWSE':
        conFiles = filesbrowse(filetypes=(('NumPy Files', '*.npz'),))
        window['CON_FILES_DISPLAY'].update([os.path.basename(f) for f in conFiles])

    elif event == 'RUN_COEX':
        if conFiles == '':
            sg.popup('No Contour Files Selected!', location=tuple(map(lambda i, j: i + j, window.CurrentLocation(), (200,160))))
            continue
        saveFile = tk.filedialog.asksaveasfilename(filetypes = (('csv files', '*.csv'),), defaultextension = (('csv files', '*.csv'),))
        if saveFile:
            threadNr += 1
            threading.Thread(target=coex, args=(window,
                                                threadNr,
                                                saveFile,
                                                conFiles,
                                                values['COEX_IMGS']),
                             daemon=True).start()

    elif event == 'THREAD_CPRINT':
        sg.cprint(values['THREAD_CPRINT'][0], c=values['THREAD_CPRINT'][1])

    elif event == 'CSV_OPEN':
        sg.popup('Could not write csv file because\ncurrent csv is open!', location=tuple(map(lambda i, j: i + j, window.CurrentLocation(), (180,160))))

plt.close()
window.close()