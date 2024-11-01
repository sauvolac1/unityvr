### This module contains basic preprocessing functions for processing the unity VR log file

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from os import mkdir, makedirs
from os.path import sep, isfile, exists
import json
import os

from unityvr.analysis import posAnalysis, shapeAnalysis


def find_files_by_extension(dirName, extension1):
    file_list = []
    for root, dirs, files in os.walk(dirName):
        for file in files:
            if file.endswith(extension1):
                file_list.append(file)
                
    #Sort files by date created
    file_list.sort(key=lambda x: os.path.getmtime(os.path.join(dirName, x)))
    return file_list

#dataframe column defs
objDfCols = ['name','collider','px','py','pz','rx','ry','rz','sx','sy','sz']
laserDfCols = ['frame', 'time', 'laser']
posDfCols = ['frame','time','x','y','angle']
slipDfCols = ['frame', 'time', 'attempted_x', 'attempted_y','attempted_angle']
ftDfCols = ['frame','ficTracTReadMs','ficTracTWriteMs','dx','dy','dz']
dtDfCols = ['frame','time','dt']
nidDfCols = ['frame','time','dt','pdsig','imgfsig']
texDfCols = ['frame','time','xtex','ytex']
vidDfCols = ['frame','time','img','duration']
# Data class definition

@dataclass
class unityVRexperiment:

    # metadata as dict
    metadata: dict

    imaging: bool = False
    brainregion: str = None

    # timeseries data
    posDf: pd.DataFrame = pd.DataFrame(columns=posDfCols)
    slipDf: pd.DataFrame = pd.DataFrame(columns=slipDfCols)
    laserDf: pd.DataFrame = pd.DataFrame(columns=laserDfCols)
    ftDf: pd.DataFrame = pd.DataFrame(columns=ftDfCols)
    nidDf: pd.DataFrame = pd.DataFrame(columns=nidDfCols)
    texDf: pd.DataFrame = pd.DataFrame(columns=texDfCols)
    vidDf: pd.DataFrame = pd.DataFrame(columns=vidDfCols)
    shapeDf: pd.DataFrame = pd.DataFrame()
    timeDf: pd.DataFrame = pd.DataFrame()

    # object locations
    objDf: pd.DataFrame = pd.DataFrame(columns=objDfCols)

    # methods
    def printMetadata(self):
        print('Metadata:\n')
        for key in self.metadata:
            print(key, ' : ', self.metadata[key])

    ## data wrangling
    def downsampleftDf(self):
        frameftDf = self.ftDf.groupby("frame").sum()
        frameftDf.reset_index(level=0, inplace=True)
        return frameftDf

    def saveData(self, saveDir, saveName):
        savepath = sep.join([saveDir,saveName,'uvr'])

        # make directory
        if not exists(savepath):
            makedirs(savepath)

        # save metadata
        with open(sep.join([savepath,'metadata.json']), 'w') as outfile:
            json.dump(self.metadata, outfile,indent=4)

        # save dataframes
        self.objDf.to_csv(sep.join([savepath,'objDf.csv']))
        self.laserDf.to_csv(sep.join([savepath,'laserDf.csv']))
        self.posDf.to_csv(sep.join([savepath,'posDf.csv']))
        self.slipDf.to_csv(sep.join([savepath,'slipDf.csv']))
        self.ftDf.to_csv(sep.join([savepath,'ftDf.csv']))
        self.nidDf.to_csv(sep.join([savepath,'nidDf.csv']))
        self.texDf.to_csv(sep.join([savepath,'texDf.csv']))
        self.vidDf.to_csv(sep.join([savepath,'vidDf.csv']))
        self.shapeDf.to_csv(sep.join([savepath,'shapeDf.csv']))
        self.timeDf.to_csv(sep.join([savepath,'timeDf.csv']))

        return savepath

# extract all dataframes from log file and save to disk
def convertJsonToPandas(dirName,fileName,saveDir, computePDtrace):

    dat = openUnityLog(dirName, fileName)
    metadat = makeMetaDict(dat, fileName)

    saveName = (metadat['expid']).split('_')[-1] + '/' + metadat['trial']
    savepath = sep.join([saveDir,saveName,'uvr'])

    # make directory
    if not exists(savepath): makedirs(savepath)

    # save metadata
    with open(sep.join([savepath,'metadata.json']), 'w') as outfile:
        json.dump(metadat, outfile,indent=4)

    # construct and save object dataframe
    objDf = objDfFromLog(dat)
    objDf.to_csv(sep.join([savepath,'objDf.csv']))

    # construct and save position and velocity dataframes
    slipDf, posDf, laserDf, ftDf, nidDf = timeseriesDfFromLog(dat, computePDtrace)
    posDf.to_csv(sep.join([savepath,'posDf.csv']))
    del posDf
    slipDf.to_csv(sep.join([savepath,'slipDf.csv']))
    del slipDf 
    laserDf.to_csv(sep.join([savepath,'laserDf.csv']))
    del laserDf 
    ftDf.to_csv(sep.join([savepath,'ftDf.csv']))
    del ftDf 
    nidDf.to_csv(sep.join([savepath,'nidDf.csv']))
    del nidDf 

    # construct and save texture dataframes
    texDf = texDfFromLog(dat)
    vidDf = vidDfFromLog(dat)
    texDf.to_csv(sep.join([savepath,'texDf.csv']))
    del texDf 
    vidDf.to_csv(sep.join([savepath,'vidDf.csv']))
    del vidDf 

    return savepath

# constructor for unityVRexperiment
def constructUnityVRexperiment(dirName,fileName):

    dat = openUnityLog(dirName, fileName)

    metadat = makeMetaDict(dat, fileName)
    objDf = objDfFromLog(dat)
    slipDf, posDf, laserDf, ftDf, nidDf = timeseriesDfFromLog(dat, computePDtrace=True)
    texDf = texDfFromLog(dat)
    vidDf = vidDfFromLog(dat)

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf, laserDf=laserDf,ftDf=ftDf,nidDf=nidDf,objDf=objDf,texDf=texDf, vidDf=vidDf, slipDf=slipDf)

    return uvrexperiment


def loadUVRData(savepath):

    with open(sep.join([savepath,'metadata.json'])) as json_file:
        metadat = json.load(json_file)
    objDf = pd.read_csv(sep.join([savepath,'objDf.csv'])).drop(columns=['Unnamed: 0'])
    # ToDo remove when Shivam has removed fixation values from posDf
    posDf = pd.read_csv(sep.join([savepath,'posDf.csv']),dtype={'fixation': 'string'}).drop(columns=['Unnamed: 0','fixation'],errors='ignore')
    ftDf = pd.read_csv(sep.join([savepath,'ftDf.csv'])).drop(columns=['Unnamed: 0'])

    try: texDf = pd.read_csv(sep.join([savepath,'texDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        texDf = pd.DataFrame()
        #No texture mapping time series was recorded with this experiment, fill with empty DataFrame

    try: vidDf = pd.read_csv(sep.join([savepath,'vidDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        vidDf = pd.DataFrame()
        #No static images were displayed, fill with empty DataFrame

    try: shapeDf = pd.read_csv(sep.join([savepath,'shapeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        shapeDf = pd.DataFrame()
        #Shape dataframe was not computed. Fill with empty DataFrame

    try: timeDf = pd.read_csv(sep.join([savepath,'timeDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        timeDf = pd.DataFrame()
        #Time dataframe was not computed. Fill with empty DataFrame
    
    try: nidDf = pd.read_csv(sep.join([savepath,'nidDf.csv'])).drop(columns=['Unnamed: 0'])
    except FileNotFoundError:
        nidDf = pd.DataFrame()
        #Nidaq dataframe may not have been extracted from the raw data due to memory/time constraints

    uvrexperiment = unityVRexperiment(metadata=metadat,posDf=posDf,laserDf=laserDf,ftDf=ftDf,nidDf=nidDf,\
                                      objDf=objDf,texDf=texDf,shapeDf=shapeDf,timeDf=timeDf, vidDf=vidDf)

    return uvrexperiment


def parseHeader(notes, headerwords, metadat):

    for i, hw in enumerate(headerwords[:-1]):
        if hw in notes:
            metadat[i] = notes[notes.find(hw)+len(hw)+1:notes.find(headerwords[i+1])].split('~')[0].strip()

    return metadat


def makeMetaDict(dat, fileName):
    headerwords = ["expid", "experiment", "genotype","flyid","sex","notes","\n"]
    metadat = ['testExp', 'test experiment', 'testGenotype', 'NA', 'NA', "NA"]

    if 'headerNotes' in dat[0].keys():
        headerNotes = dat[0]['headerNotes']
        metadat = parseHeader(headerNotes, headerwords, metadat)

    [datestr, timestr] = fileName.split('.')[0].split('_')[1:3]

    matching = [s for s in dat if "ficTracBallRadius" in s]
    if len(matching) == 0:
        print('no fictrac metadata')
        ballRad = 0.0
    else:
        ballRad = matching[0]["ficTracBallRadius"]

    matching = [s for s in dat if "refreshRateHz" in s]
    setFrameRate = matching[0]["refreshRateHz"]

    metadata = {
        'expid': metadat[0],
        'experiment': metadat[1],
        'genotype': metadat[2],
        'sex': metadat[4],
        'flyid': metadat[3],
        'trial': 'trial'+fileName.split('.')[0].split('_')[-1][1:],
        'date': datestr,
        'time': timestr,
        'ballRad': ballRad,
        'setFrameRate': setFrameRate,
        'notes': metadat[5],
        'angle_convention':"right-handed"
    }

    return metadata


def openUnityLog(dirName, fileName):
    '''load json log file'''
    import json
    from os.path import sep

    # Opening JSON file
    f = open(sep.join([dirName, fileName]))

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    return data


# Functions for extracting data from log file and converting it to pandas dataframe

def objDfFromLog(dat):
    # get dataframe with info about objects in vr
    matching = [s for s in dat if "meshGameObjectPath" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'name': match['meshGameObjectPath'],
                    'collider': match['colliderType'],
                    'px': match['worldPosition']['x'],
                    'py': match['worldPosition']['z'],
                    'pz': match['worldPosition']['y'],
                    'rx': match['worldRotationDegs']['x'],
                    'ry': match['worldRotationDegs']['z'],
                    'rz': match['worldRotationDegs']['y'],
                    'sx': match['worldScale']['x'],
                    'sy': match['worldScale']['z'],
                    'sz': match['worldScale']['y']}
        entries[entry] = pd.Series(framedat).to_frame().T
    if len(entries) > 0:
        return pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()


def posDfFromLog(dat):
    # get info about camera position in vr
    collision = np.any(np.array(['attemptedTranslation' in dat[i] for i in range(len(dat))]))
    if not collision:
        matching = [s for s in dat if "worldPosition" in s]
        entries = [None]*len(matching)
        for entry, match in enumerate(matching):
            framedat = {'frame': match['frame'],
                            'time': match['timeSecs'],
                            'x': match['worldPosition']['x'],
                            'y': match['worldPosition']['z'], #axes are named differently in Unity
                            'angle': (match['worldRotationDegs']['y']), #flip due to left handed convention in Unity
                           }
            entries[entry] = pd.Series(framedat).to_frame().T
    
    else:     
        matching = [s for s in dat if "attemptedTranslation" in s]
        entries = [None]*len(matching)
        for entry, match in enumerate(matching):
            framedat = {'frame': match['frame'],
                            'time': match['timeSecs'],
                            'x': match['worldPosition']['x'],
                            'y': match['worldPosition']['z'], #axes are named differently in Unity
                            #'angle': match['worldRotationDegs']['y'],
                            'angle': (match['worldRotationDegs']['y']), #flip due to left handed convention in Unity
                            'dx':match['actualTranslation']['x'],
                            'dy':match['actualTranslation']['z'],
                            'dxattempt': match['attemptedTranslation']['x'],
                            'dyattempt': match['attemptedTranslation']['z'],
                           }
            entries[entry] = pd.Series(framedat).to_frame().T
    print('correcting for Unity angle convention.')

    if len(entries) > 0:
        return  pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()
    
def slipDfFromLog(dat):
    matching = [s for s in dat if "fictracAttempt" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
            fictracdat = {'frame': match['frame'],
                            'time': match['timeSecs'],
                            'attempted_x': match['fictracAttempt']['x'],
                            'attempted_y': match['fictracAttempt']['y'],
                            'attempted_angle': match['fictracAttempt']['z']%360
                            }
            entries[entry] = pd.Series(fictracdat).to_frame().T
    if len(entries) > 0:
        return  pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()

def laserDfFromLog(dat):
    matching = [s for s in dat if "VoltageWritten" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'laser': match['VoltageWritten']
                       }
        entries[entry] = pd.Series(framedat).to_frame().T
    if len(entries) > 0:
        return  pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()


def ftDfFromLog(dat):
    # get fictrac data
    matching = [s for s in dat if "ficTracDeltaRotationVectorLab" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                        'ficTracTReadMs': match['ficTracTimestampReadMs'],
                        'ficTracTWriteMs': match['ficTracTimestampWriteMs'],
                        'dx': match['ficTracDeltaRotationVectorLab']['x'],
                        'dy': match['ficTracDeltaRotationVectorLab']['y'],
                        'dz': match['ficTracDeltaRotationVectorLab']['z']}
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        return pd.concat(entries, ignore_index = True)
    else:
        return pd.DataFrame()


def dtDfFromLog(dat):
    # get delta time info
    matching = [s for s in dat if "deltaTime" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'dt': match['deltaTime']}
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        return pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()


def pdDfFromLog(dat, computePDtrace):
    # get NiDaq signal
    matching = [s for s in dat if "tracePD" in s]
    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        if computePDtrace:
            framedat = {'frame': match['frame'],
                        'time': match['timeSecs'],
                        'pdsig': match['tracePD'],
                        'imgfsig': match['imgFrameTrigger'],
                        }
        else:
            framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'imgfsig': match['imgFrameTrigger'],
                    }
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        if computePDtrace:
            pdDf = pd.concat(entries,ignore_index = True)[['frame', 'time', 'pdsig', 'imgfsig']].drop_duplicates()
        else:
            pdDf = pd.concat(entries,ignore_index = True)[['frame', 'time','imgfsig']].drop_duplicates()
        return pdDf
    else:
        return pd.DataFrame()


def texDfFromLog(dat):
    # get texture remapping log
    matching = [s for s in dat if "xpos" in s]
    if len(matching) == 0: return pd.DataFrame()

    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        if 'ypos' in match:
            framedat = {'frame': match['frame'],
                        'time': match['timeSecs'],
                        'xtex': match['xpos'],
                        'ytex': match['ypos']}
        else:
            framedat = {'frame': match['frame'],
                        'time': match['timeSecs'],
                        'xtex': match['xpos'],
                        'ytex': 0}
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        texDf = pd.concat(entries,ignore_index = True)
        texDf.time = texDf.time-texDf.time[0]
        return texDf
    else:
        return pd.DataFrame()


def vidDfFromLog(dat):
    # get static image presentations and times
    matching = [s for s in dat if "backgroundTextureNowInUse" in s]
    if len(matching) == 0: return pd.DataFrame()

    entries = [None]*len(matching)
    for entry, match in enumerate(matching):
        framedat = {'frame': match['frame'],
                    'time': match['timeSecs'],
                    'img': match['backgroundTextureNowInUse'].split('/')[-1],
                    'duration': match['durationSecs']}
        entries[entry] = pd.Series(framedat).to_frame().T

    if len(entries) > 0:
        return pd.concat(entries,ignore_index = True)
    else:
        return pd.DataFrame()


def ftTrajDfFromLog(directory, filename):
    cols = [14,15,16,17,18]
    colnames = ['x','y','heading','travelling','speed']
    ftTrajDf = pd.read_csv(directory+"/"+filename,usecols=cols,names=colnames)
    return ftTrajDf

def timeseriesDfFromLog(dat, computePDtrace):
    from scipy.signal import medfilt

    posDf = pd.DataFrame(columns=posDfCols)
    slipDf = pd.DataFrame(columns=slipDfCols)
    laserDf = pd.DataFrame(columns=laserDfCols)
    ftDf = pd.DataFrame(columns=ftDfCols)
    dtDf = pd.DataFrame(columns=dtDfCols)
    if computePDtrace:
        pdDf = pd.DataFrame(columns = ['frame','time','pdsig', 'imgfsig'])
    else:
        pdDf = pd.DataFrame(columns = ['frame','time', 'imgfsig'])

    posDf = posDfFromLog(dat)
    slipDf = slipDfFromLog(dat)
    laserDf = laserDfFromLog(dat)
    ftDf = ftDfFromLog(dat)
    dtDf = dtDfFromLog(dat)

    try: pdDf = pdDfFromLog(dat, computePDtrace)
    except: print("No analog input data was recorded.")

    if len(posDf) > 0: posDf.time = posDf.time-posDf.time[0]
    if len(slipDf) > 0: slipDf.time = slipDf.time-slipDf.time[0]
    if len(laserDf) > 0: laserDf.time = laserDf.time-laserDf.time[0]
    if len(dtDf) > 0: dtDf.time = dtDf.time-dtDf.time[0]
    if len(pdDf) > 0: pdDf.time = pdDf.time-pdDf.time[0]

    if len(ftDf) > 0:
        ftDf.ficTracTReadMs = ftDf.ficTracTReadMs-ftDf.ficTracTReadMs[0]
        ftDf.ficTracTWriteMs = ftDf.ficTracTWriteMs-ftDf.ficTracTWriteMs[0]
    else:
        print("No fictrac signal was recorded.")

    if len(dtDf) > 0: posDf = pd.merge(dtDf, posDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)

    if len(pdDf) > 0 and len(dtDf) > 0:
        nidDf = pd.merge(dtDf, pdDf, on="frame", how='outer').rename(columns={'time_x':'time'}).drop(['time_y'],axis=1)
        if computePDtrace:
            nidDf["pdFilt"]  = nidDf.pdsig.values
            nidDf.pdFilt.values[np.isfinite(nidDf.pdsig.values)] = medfilt(nidDf.pdsig.values[np.isfinite(nidDf.pdsig.values)])
            nidDf["pdThresh"]  = 1*(np.asarray(nidDf.pdFilt>=np.nanmedian(nidDf.pdFilt.values)))

        nidDf["imgfFilt"]  = nidDf.imgfsig.values
        nidDf.imgfFilt.values[np.isfinite(nidDf.imgfsig.values)] = medfilt(nidDf.imgfsig.values[np.isfinite(nidDf.imgfsig.values)])
        nidDf["imgfThresh"]  = 1*(np.asarray(nidDf.imgfFilt.values>=np.nanmedian(nidDf.imgfFilt.values))).astype(np.int8)

        nidDf = generateInterTime(nidDf)
    else:
        nidDf = pd.DataFrame()

    return slipDf, posDf, laserDf, ftDf, nidDf


def generateInterTime(tsDf):
    from scipy import interpolate

    tsDf['framestart'] = np.hstack([0,1*np.diff(tsDf.time)>0])
    #tsDf['framestart'] = tsDf.framestart.astype(bool)

    tsDf['counts'] = 1
    #tsDf['counts'] = tsDf.counts.astype(np.int8)
    sampperframe = tsDf.groupby('frame').sum()[['time','dt','counts']].reset_index(level=0)
    sampperframe['fs'] = sampperframe.counts/sampperframe.dt

    frameStartIndx = np.hstack((0,np.where(tsDf.framestart)[0]))
    frameStartIndx = np.hstack((frameStartIndx, frameStartIndx[-1]+sampperframe.counts.values[-1]-1))
    frameIndx = tsDf.index.values
    del sampperframe

    frameNums = tsDf.frame[frameStartIndx].values.astype('int')
    timeAtFramestart = tsDf.time[frameStartIndx].values

    #generate interpolated frames
    frameinterp_f = interpolate.interp1d(frameStartIndx,frameNums)
    tsDf['frameinterp'] = frameinterp_f(frameIndx)

    timeinterp_f = interpolate.interp1d(frameStartIndx,timeAtFramestart)
    tsDf['timeinterp'] = timeinterp_f(frameIndx)

    return tsDf


def mergeDfs(dirName, unityFiles, plot=False, rotate_by=None, derive=True, kinematic_subject=False):
    fullDf = pd.DataFrame()
    fullVidDf = pd.DataFrame()
    for i,f in enumerate(unityFiles):
        print(f)
        uvrTest = constructUnityVRexperiment(dirName,f)
        posDf = posAnalysis.position(uvrTest, 
                         rotate_by = rotate_by,
                         derive = derive, #derive set to true adds 
                         #derived parameters like velocity and angle to the dataframe
                         plot=plot
                         #pass the following parameters to save the dataframe in a chosen directory
                         #,plotsave=False,saveDir=saveDir
                        )

         #account for weird example kinematic effect
        if kinematic_subject: 
            if len(posDf['time'].unique()) != len(posDf):
                posDf = posDf.iloc[::2].reset_index(drop=True)
            posDf = posDf.fillna(0)
        
        vidDf = uvrTest.vidDf.copy()
        laserDf = pd.merge(posDf, uvrTest.laserDf, on=['frame']).copy()
        laserDf = laserDf.rename(columns={"time_x": "timeInTrial", "time_y": "time"})
        laserDf['trial'] = i
        laserDf['trial_name'] = f

        #ASSUMPTION: UNITY X AND Y START AT 0, if problem fixed on unity end, remove snippet from below
        #relies on sorting by time
        if 'x' in fullDf:
            laserDf['x'] = laserDf['x'] + fullDf['x'].iloc[-1]
            laserDf['y'] = laserDf['y'] + fullDf['y'].iloc[-1]
            laserDf['frame'] = laserDf['frame'] + fullDf['frame'].iloc[-1]
            laserDf['time'] = laserDf['time'] + fullDf['time'].iloc[-1]
            if derive:
                laserDf['s'] = laserDf['s'] + fullDf['s'].iloc[-1]
            
            #breaks independance
            vidDf['frame'] = vidDf['frame'] + fullDf['frame'].iloc[-1]

        fullDf = pd.concat([fullDf,laserDf])
        fullVidDf = pd.concat([fullVidDf,vidDf])
    return fullDf.reset_index(drop=True), fullVidDf.reset_index(drop = True)
