from math import pi
from imghdr import what
import os
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
def scc(window,threadNr,saveFile,imFiles,mode,threshOnly,writeImgs,fluorescent,size,minArea,maxArea,circularityThresh,offset,channel):
    window.write_event_value('THREAD_CPRINT', (str(mode)+' '+str(threadNr)+' running...', 'black on yellow'))
    
    counts = []
    backgrounds = []
    spreads = []
    areas = []
    csv = []
    concsv = []
    
    #Min/Max Areas, Brush
    oddSize = size + (1 if size%2==0 else 0)
    threshKernel = 4*size + (1 if (4*size)%2==0 else 0)
    morphKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    
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
        images[5] = cv2.imread(baseImage)
        images[0] = cv2.cvtColor(images[5], cv2.COLOR_BGR2GRAY)
        if fluorescent:
            images[0] = cv2.bitwise_not(images[0])

        #Truncate pixel values to mode pixel value
        pxList, pxHist =  np.unique(images[0], return_counts=True)
        pxMode = pxList[np.argmax(pxHist[5:-5])]    #Ignore brightest and darkest values, in case of ceiling/floor effect
        images[1] = cv2.GaussianBlur(cv2.threshold(images[0], pxMode, 255, cv2.THRESH_TRUNC)[1], (oddSize,oddSize), 0)

        #Apply adaptive threshold to image and lightly blur to supress noise
        images[2] = cv2.medianBlur(cv2.adaptiveThreshold(images[1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, threshKernel, offset), 5)
        
        #Assess background brightness
        ptOtsu = cv2.threshold(images[1], 0, 255, cv2.THRESH_OTSU)[0]
        
        #Record background and offset used
        if mode == 'Batch':
            backgrounds.append(pxMode)
            spreads.append(pxMode-ptOtsu)
            if threshOnly:
                counts.append('N/A')
                areas.append('N/A')
                continue
        
        #Process image
        images[3] = cv2.morphologyEx(images[2], cv2.MORPH_OPEN, morphKernel)
        images[4] = cv2.morphologyEx(images[3], cv2.MORPH_CLOSE, morphKernel)

        contours[0] = cv2.findContours(images[4], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        #Area filter
        contours[1] = [e for e in contours[0] if cv2.contourArea(e) > minArea and cv2.contourArea(e) < maxArea]

        #Circularity filter
        contours[2] = [e for e in contours[1] if (cv2.moments(e)['m00']**2)/(2*pi * (cv2.moments(e)['mu02'] + cv2.moments(e)['mu20'])) > circularityThresh]

        if mode == 'Batch':
            counts.append(len(contours[2]))
            areas.append(np.mean([cv2.contourArea(e) for e in contours[2]]))
            
        #Image Generation
        if mode == 'Preview' or writeImgs:
            #Draw contours
            cv2.drawContours(images[5], contours[0], -1, (200,0,255), 2)
            cv2.drawContours(images[5], contours[1], -1, (0,255,255), 2)
            cv2.drawContours(images[5], contours[2], -1, (200,220,0), 2)

            #Notate counts
            cv2.putText(images[5], '!Area:'+str(len(contours[0])-len(contours[1])), (8,36), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,0,255), 2)
            cv2.putText(images[5], '!Circular:'+str(len(contours[1])-len(contours[2])), (8,72), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(images[5], 'Cells:'+str(len(contours[2])), (8,108), cv2.FONT_HERSHEY_SIMPLEX, 1, (200,220,0), 2)
        
        if mode == 'Batch':
            if writeImgs:
                outputImg = '_$$$.'.join(baseImage.rsplit('.',1))
                cv2.imwrite(outputImg, images[5])
            if channel:
                concsv.append(np.array(contours[3]))
        
    if mode == 'Batch':
        #csv Generation
        csv = pd.DataFrame({'image':imgsToProcess,'count':counts,'background':backgrounds,'spread':spreads,'averageArea':areas})
        csv = csv[['image', 'count', 'background','spread', 'averageArea']]

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

    #Coex analysis
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
            cv2.putText(image, 'Colabeled Cells:'+str(len(imgCoContours)), (8,144), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
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
    elif(elementID) == 'SelStr': return 'How strongly cells must stand out from surrounding tissue.'
    elif(elementID) == 'Fluorescent': return 'Enable if analyzing fluorescent images.'
    elif(elementID) == 'Size': return 'Approximate radius, in pixels,\nof smallest cells'
    elif(elementID) == 'Minimum Area': return 'Minimum area of an outlined cell\nto pass the size filter.'
    elif(elementID) == 'Maximum Area': return 'Maximum area of an outlined cell\nto pass the size filter.'
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
    for k in list(values.keys())[5:12]:
        v = config.get('GENERAL', k)
        window[k].update(True if v=='True' else False if v=='False' else v)

def exportcfg(file,values_dict):
    for k in list(values_dict.keys())[5:12]:
        config.set('GENERAL', k, str(values_dict[k]))
    with open(file, 'w') as configfile:
        config.write(configfile)
    
def stringtobool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False

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
minArea = 100
maxArea = 600
circularityThresh = 0.8
offset = 12

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

        [sg.T('Selection Strength', tooltip=tt('SelStr'), key='SEL_STR')],
                 [sg.Spin([i/10 for i in range(0,501)], initial_value=offset, size=(6,1), key='OFFSET')],
        [sg.T('', size=(None,1))]]

    second_col = [
        [sg.Checkbox('Fluorescent', default=fluorescent, tooltip=tt('Fluorescent'), key='FLUO')],
        [sg.T('Size', tooltip=tt('Size')),
         sg.Spin([i for i in range(1,100)], initial_value=size, size=(6,1), key='SIZE')],
        [sg.T('Minimum Area', tooltip=tt('Minimum Area')),
         sg.Spin([i for i in range(0,4000)], initial_value=minArea, size=(6,1), key='MIN_AREA')],
        [sg.T('Maximum Area', tooltip=tt('Maximum Area')),
         sg.Spin([i for i in range(0,10000)], initial_value=maxArea, size=(6,1), key='MAX_AREA')],
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
            importcfg(cfgFile)
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
                                               float(values['MIN_AREA']),
                                               float(values['MAX_AREA']),
                                               float(values['CIRCLE_THR']),
                                               float(values['OFFSET']),
                                               values['CH_ID']),
                             daemon=True).start()

    elif event == 'PREVIEW_RETURN':
        fig = plt.figure(figsize=(32, 24))
        labels = ['Source','Truncated','After threshold','After gap filling','After size filtering','Final']
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