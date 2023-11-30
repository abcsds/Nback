#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on noviembre 30, 2023, at 15:50
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_welcome
import numpy as np
import serial
import psychopy.logging as log
import string
import serial
import time
from pylsl import StreamInfo, StreamOutlet
import tobii_research as tr

# Setup LSL outlet
log.info("Creating LSL streams")
info = StreamInfo("Psychopy", "Markers", 1, 0, "string", "PsychopyUUID0001")
outlet = StreamOutlet(info)
outlet.push_sample(["begin_experiment"])

info_tobii = StreamInfo('Pupilometry', 'Pupilometry', 2, 120, 'float32', 'myuidw43536')
outlet_tobii = StreamOutlet(info_tobii)

# Setup serial connection to arduino
# Used for ttl triggers to EEG amp
# arduino = serial.Serial("COM5", 9600)

# Setup eyetracker
EYETRACKER_ADDRESS = "tobii-prp://TPFC1-010202448011"
log.info(f"Connecting to tobii device: {EYETRACKER_ADDRESS}")
et = tr.EyeTracker(EYETRACKER_ADDRESS)

log.info(f"Eyetracker: {et}")

def gaze_data_callback(gaze_data):
#    if gaze_data["left_pupil_validity"]:
#        log.data(f"Pupil diameter: L {gaze_data['left_pupil_diameter']}")
#    if gaze_data["right_pupil_validity"]:
#        log.data(f"Pupil diameter: R {gaze_data['right_pupil_diameter']}")
    outlet_tobii.push_sample([gaze_data['left_pupil_diameter'], gaze_data['right_pupil_diameter']])

# Experiment Vars
stimulus_duration = 3
p_target = 0.2 # Probability of stimulus being target
letter = ""
all_letters = list(string.ascii_uppercase)
all_letters.remove("N") # Testers got confused with the letter N
isi = 0.5 # Inter-stimulus-interval
# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'nback'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Dell 1\\Documents\\Nback\\nback_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    text_welcome = visual.TextStim(win=win, name='text_welcome',
        text='¡Te damos la bienvenida!\n\nA continuación llevaremos a cabo algunas calibraciones, y te mostraremos la tarea NBack.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_welcome = keyboard.Keyboard()
    # Run 'Begin Experiment' code from code_welcome
    #for i in range(10): # Marks start of experiment
    #    # arduino.write(b"p")
    #    outlet.push_sample([f"sync_{i+1}"])
    #    time.sleep(0.5)
    
    # Start the eyetracker
    et.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_data_callback, as_dictionary=True)
    
    win.mouseVisible = False
    
    # --- Initialize components for Routine "intro_cal" ---
    text_intro_cal = visual.TextStim(win=win, name='text_intro_cal',
        text='Te mostrarémos unos puntos en la pantalla.\nPor favor, fija tu mirada en ellos. \nCon esto calibrarémos el escaner ocular.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_cal = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fix" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_nothing = visual.TextStim(win=win, name='text_nothing',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "calibration" ---
    circle_cal_1 = visual.ShapeStim(
        win=win, name='circle_cal_1',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(-.711, 0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    circle_cal_2 = visual.ShapeStim(
        win=win, name='circle_cal_2',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0.711, 0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    circle_cal_3 = visual.ShapeStim(
        win=win, name='circle_cal_3',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(-0.711, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    circle_cal_4 = visual.ShapeStim(
        win=win, name='circle_cal_4',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0.711, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    circle_cal_5 = visual.ShapeStim(
        win=win, name='circle_cal_5',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "intro" ---
    text_intro = visual.TextStim(win=win, name='text_intro',
        text='Tarea NBack:\nSe mostrarán una secuencia de letras.\nPresiona la barra espaciadora si la letra presentada es igual a la letra presentada cierto numero de letras antes. A ese número le llamamos N.\n\nN es diferente en cada bloque.\nAntes de empezar verás unos ejemplos.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro = keyboard.Keyboard()
    
    # --- Initialize components for Routine "intro_ex1" ---
    text_intro_ex1 = visual.TextStim(win=win, name='text_intro_ex1',
        text='A continuación se mostrará un ejemplo para la tarea 1-back\nPresiona la barra espaciadora cuando la letra presentada sea igual a la última letra.\n\nEl valor de N es 1.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_ex1 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ex1" ---
    text_ex1_1 = visual.TextStim(win=win, name='text_ex1_1',
        text='X',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rect_ex1_1 = visual.Rect(
        win=win, name='rect_ex1_1',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-1.0, interpolate=True)
    text_ex1_2 = visual.TextStim(win=win, name='text_ex1_2',
        text='B',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    rect_ex1_2 = visual.Rect(
        win=win, name='rect_ex1_2',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-3.0, interpolate=True)
    text_ex1_3 = visual.TextStim(win=win, name='text_ex1_3',
        text='H',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    rect_ex1_3 = visual.Rect(
        win=win, name='rect_ex1_3',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-5.0, interpolate=True)
    text_ex1_4 = visual.TextStim(win=win, name='text_ex1_4',
        text='H',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    rect_ex1_4 = visual.Rect(
        win=win, name='rect_ex1_4',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='transparent',
        opacity=None, depth=-7.0, interpolate=True)
    text_ex1_5 = visual.TextStim(win=win, name='text_ex1_5',
        text='A',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    rect_ex1_5 = visual.Rect(
        win=win, name='rect_ex1_5',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-9.0, interpolate=True)
    text_ex1_6 = visual.TextStim(win=win, name='text_ex1_6',
        text='X',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    rect_ex1_6 = visual.Rect(
        win=win, name='rect_ex1_6',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-11.0, interpolate=True)
    text_ex1_array = visual.TextStim(win=win, name='text_ex1_array',
        text='X  B  H  H  A  X',
        font='Open Sans',
        pos=(0, -.40), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    key_resp_ex1 = keyboard.Keyboard()
    text_ex1_cue = visual.TextStim(win=win, name='text_ex1_cue',
        text='¡Presiona ahora!',
        font='Open Sans',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "intro_ex2" ---
    text_intro_ex2 = visual.TextStim(win=win, name='text_intro_ex2',
        text='A continuación se mostrara un ejemplo para la tarea 2-back\nPresiona la barra espaciadora cuando la letra presentada sea igual a la penúltima letra.\n\nEl valor de N es 2.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_ex2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ex2" ---
    text_ex2_1 = visual.TextStim(win=win, name='text_ex2_1',
        text='X',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rect_ex2_1 = visual.Rect(
        win=win, name='rect_ex2_1',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-1.0, interpolate=True)
    text_ex2_2 = visual.TextStim(win=win, name='text_ex2_2',
        text='B',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    rect_ex2_2 = visual.Rect(
        win=win, name='rect_ex2_2',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-3.0, interpolate=True)
    text_ex2_3 = visual.TextStim(win=win, name='text_ex2_3',
        text='A',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    rect_ex2_3 = visual.Rect(
        win=win, name='rect_ex2_3',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-5.0, interpolate=True)
    text_ex2_4 = visual.TextStim(win=win, name='text_ex2_4',
        text='B',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    rect_ex2_4 = visual.Rect(
        win=win, name='rect_ex2_4',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='transparent',
        opacity=None, depth=-7.0, interpolate=True)
    text_ex2_5 = visual.TextStim(win=win, name='text_ex2_5',
        text='H',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    rect_ex2_5 = visual.Rect(
        win=win, name='rect_ex2_5',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-9.0, interpolate=True)
    text_ex2_6 = visual.TextStim(win=win, name='text_ex2_6',
        text='U',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    rect_ex2_6 = visual.Rect(
        win=win, name='rect_ex2_6',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-11.0, interpolate=True)
    text_ex2_array = visual.TextStim(win=win, name='text_ex2_array',
        text='X  B  A  B  H  U',
        font='Open Sans',
        pos=(0, -.40), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    key_resp_ex2 = keyboard.Keyboard()
    text_ex2_cue = visual.TextStim(win=win, name='text_ex2_cue',
        text='¡Presiona ahora!',
        font='Open Sans',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    
    # --- Initialize components for Routine "intro_ex3" ---
    text_intro_ex3 = visual.TextStim(win=win, name='text_intro_ex3',
        text='A continuación se mostrará un ejemplo para la tarea 3-back\nPresiona la barra espaciadora cuando la letra presentada sea igual a la ante-penúltima letra.\n\nEl valor de N es 3.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_ex3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "ex3" ---
    text_ex3_1 = visual.TextStim(win=win, name='text_ex3_1',
        text='X',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    rect_ex3_1 = visual.Rect(
        win=win, name='rect_ex3_1',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-1.0, interpolate=True)
    text_ex3_2 = visual.TextStim(win=win, name='text_ex3_2',
        text='U',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    rect_ex3_2 = visual.Rect(
        win=win, name='rect_ex3_2',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-3.0, interpolate=True)
    text_ex3_3 = visual.TextStim(win=win, name='text_ex3_3',
        text='M',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    rect_ex3_3 = visual.Rect(
        win=win, name='rect_ex3_3',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(-0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='transparent',
        opacity=None, depth=-5.0, interpolate=True)
    text_ex3_4 = visual.TextStim(win=win, name='text_ex3_4',
        text='X',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    rect_ex3_4 = visual.Rect(
        win=win, name='rect_ex3_4',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.03, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='transparent',
        opacity=None, depth=-7.0, interpolate=True)
    text_ex3_5 = visual.TextStim(win=win, name='text_ex3_5',
        text='A',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    rect_ex3_5 = visual.Rect(
        win=win, name='rect_ex3_5',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.09, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='red', fillColor='transparent',
        opacity=None, depth=-9.0, interpolate=True)
    text_ex3_6 = visual.TextStim(win=win, name='text_ex3_6',
        text='M',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    rect_ex3_6 = visual.Rect(
        win=win, name='rect_ex3_6',
        width=(0.08, 0.1)[0], height=(0.08, 0.1)[1],
        ori=0.0, pos=(0.16, -.40), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='green', fillColor='transparent',
        opacity=None, depth=-11.0, interpolate=True)
    text_ex3_array = visual.TextStim(win=win, name='text_ex3_array',
        text='X  U  M  X  A  M',
        font='Open Sans',
        pos=(0, -.40), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    key_resp_ex3 = keyboard.Keyboard()
    text_ex3_cue2 = visual.TextStim(win=win, name='text_ex3_cue2',
        text='¡Presiona ahora!',
        font='Open Sans',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    text_ex3_cue1 = visual.TextStim(win=win, name='text_ex3_cue1',
        text='¡Presiona ahora!',
        font='Open Sans',
        pos=(0, -0.3), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-15.0);
    
    # --- Initialize components for Routine "intro_train1" ---
    text_intro_train1 = visual.TextStim(win=win, name='text_intro_train1',
        text='En seguida entrenarás en la tarea N-back con N =1.\nEl entrenamiento termina cuando consigas 5 aciertos.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_train1 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "train1" ---
    text_train1 = visual.TextStim(win=win, name='text_train1',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_train1 = keyboard.Keyboard()
    text_train1_feedback = visual.TextStim(win=win, name='text_train1_feedback',
        text='',
        font='Open Sans',
        pos=(0, -.4), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "intro_train2" ---
    text_intro_train2 = visual.TextStim(win=win, name='text_intro_train2',
        text='En seguida entrenarás en la tarea N-back con N =2.\nEl entrenamiento termina cuando consigas 5 aciertos.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_intro_train2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "train2" ---
    text_train2 = visual.TextStim(win=win, name='text_train2',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_train2 = keyboard.Keyboard()
    text_train2_feedback = visual.TextStim(win=win, name='text_train2_feedback',
        text='',
        font='Open Sans',
        pos=(0, -.4), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "intro_nasatlx" ---
    text_intro_nasatlx = visual.TextStim(win=win, name='text_intro_nasatlx',
        text='Al terminar cada bloque, se te pedirá que evalues la tarea en seis diferentes escalas:',
        font='Open Sans',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_intro_nasatlx_1 = visual.TextStim(win=win, name='text_intro_nasatlx_1',
        text='- Exigencia Mental: ¿Cuánta actividad mental necesitaste?\n- Exigencia Física: ¿Cuánta actividad física fue necesaria?\n- Exigencia Temporal: ¿Cuanta presión de tiempo sentiste?\n',
        font='Open Sans',
        pos=(-0.4, 0), height=0.04, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_intro_nasatlx_2 = visual.TextStim(win=win, name='text_intro_nasatlx_2',
        text='- Esfuerzo: ¿Qué tanto esfuerzo pusiste en la tarea?\n- Rendimiento: ¿Cómo evaluas tu rendimiento en la tarea?\n- Nivel de Frustración: ¿Qué tanta frustración te produjo la tarea?',
        font='Open Sans',
        pos=(0.4, 0), height=0.04, wrapWidth=0.5, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    text_intro_nasatlx_3 = visual.TextStim(win=win, name='text_intro_nasatlx_3',
        text='Evalua dando click en la barra debajo de cada escala.\nA continuación se mostrará un ejemplo que no se evaluará. Es simplemente para familiarizarte con el instrumento.\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, -0.37), height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_resp_intro_nasatlx = keyboard.Keyboard()
    
    # --- Initialize components for Routine "nasaTLX" ---
    text_mental_demand = visual.TextStim(win=win, name='text_mental_demand',
        text='Exigencia Mental',
        font='Open Sans',
        pos=(-.4, .4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_mental_demand = visual.Slider(win=win, name='slider_mental_demand',
        startValue=None, size=(0.6, 0.02), pos=(-.4, .3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    text_physical_demand = visual.TextStim(win=win, name='text_physical_demand',
        text='Exigencia Física',
        font='Open Sans',
        pos=(-.4, .1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    slider_physical_demand = visual.Slider(win=win, name='slider_physical_demand',
        startValue=None, size=(0.6, 0.02), pos=(-0.4, .0), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    text_temporal_demand = visual.TextStim(win=win, name='text_temporal_demand',
        text='Exigencia Temporal',
        font='Open Sans',
        pos=(-0.4, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    slider_temporal_demand = visual.Slider(win=win, name='slider_temporal_demand',
        startValue=None, size=(0.6, 0.02), pos=(-0.4, -0.3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    text_effort = visual.TextStim(win=win, name='text_effort',
        text='Esfuerzo',
        font='Open Sans',
        pos=(.4, .4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    slider_effort = visual.Slider(win=win, name='slider_effort',
        startValue=None, size=(0.6, 0.02), pos=(.4, .3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-7, readOnly=False)
    text_performance = visual.TextStim(win=win, name='text_performance',
        text='Rendimiento',
        font='Open Sans',
        pos=(.4, .1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    slider_performance = visual.Slider(win=win, name='slider_performance',
        startValue=None, size=(0.6, 0.02), pos=(0.4, .0), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-9, readOnly=False)
    text_frustration = visual.TextStim(win=win, name='text_frustration',
        text='Frustración',
        font='Open Sans',
        pos=(0.4, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    slider_frustration = visual.Slider(win=win, name='slider_frustration',
        startValue=None, size=(0.6, 0.02), pos=(0.4, -0.3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-11, readOnly=False)
    text_nasatlx = visual.TextStim(win=win, name='text_nasatlx',
        text='[Presiona barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, -0.45), height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    key_resp_nasatlx = keyboard.Keyboard()
    
    # --- Initialize components for Routine "pretrial" ---
    text_pretrial = visual.TextStim(win=win, name='text_pretrial',
        text='¡Ahora empezaremos con el experimento!\n\nResponderas tareas con diferente N. El valor de N se mostrara al principio de cada bloque, pero no podras ver las letras anteriores o siguientes.\n\nContinúa si las instrucciones han quedado claras.\n\n[Presiona la barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_pretrial = keyboard.Keyboard()
    
    # --- Initialize components for Routine "displayn" ---
    text_displayn = visual.TextStim(win=win, name='text_displayn',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_displayn = keyboard.Keyboard()
    
    # --- Initialize components for Routine "fix" ---
    cross = visual.ShapeStim(
        win=win, name='cross', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text_nothing = visual.TextStim(win=win, name='text_nothing',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "trial" ---
    text_letter = visual.TextStim(win=win, name='text_letter',
        text='',
        font='Open Sans',
        pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_trial = keyboard.Keyboard()
    
    # --- Initialize components for Routine "nasaTLX" ---
    text_mental_demand = visual.TextStim(win=win, name='text_mental_demand',
        text='Exigencia Mental',
        font='Open Sans',
        pos=(-.4, .4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_mental_demand = visual.Slider(win=win, name='slider_mental_demand',
        startValue=None, size=(0.6, 0.02), pos=(-.4, .3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    text_physical_demand = visual.TextStim(win=win, name='text_physical_demand',
        text='Exigencia Física',
        font='Open Sans',
        pos=(-.4, .1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    slider_physical_demand = visual.Slider(win=win, name='slider_physical_demand',
        startValue=None, size=(0.6, 0.02), pos=(-0.4, .0), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    text_temporal_demand = visual.TextStim(win=win, name='text_temporal_demand',
        text='Exigencia Temporal',
        font='Open Sans',
        pos=(-0.4, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    slider_temporal_demand = visual.Slider(win=win, name='slider_temporal_demand',
        startValue=None, size=(0.6, 0.02), pos=(-0.4, -0.3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-5, readOnly=False)
    text_effort = visual.TextStim(win=win, name='text_effort',
        text='Esfuerzo',
        font='Open Sans',
        pos=(.4, .4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    slider_effort = visual.Slider(win=win, name='slider_effort',
        startValue=None, size=(0.6, 0.02), pos=(.4, .3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-7, readOnly=False)
    text_performance = visual.TextStim(win=win, name='text_performance',
        text='Rendimiento',
        font='Open Sans',
        pos=(.4, .1), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    slider_performance = visual.Slider(win=win, name='slider_performance',
        startValue=None, size=(0.6, 0.02), pos=(0.4, .0), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-9, readOnly=False)
    text_frustration = visual.TextStim(win=win, name='text_frustration',
        text='Frustración',
        font='Open Sans',
        pos=(0.4, -0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    slider_frustration = visual.Slider(win=win, name='slider_frustration',
        startValue=None, size=(0.6, 0.02), pos=(0.4, -0.3), units=win.units,
        labels=("Baja", "", "Alta"), ticks=(-10, 0, 10), granularity=0.0,
        style='rating', styleTweaks=('triangleMarker',), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-11, readOnly=False)
    text_nasatlx = visual.TextStim(win=win, name='text_nasatlx',
        text='[Presiona barra espaciadora para continuar...]',
        font='Open Sans',
        pos=(0, -0.45), height=0.02, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    key_resp_nasatlx = keyboard.Keyboard()
    
    # --- Initialize components for Routine "end" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='Este es el fin del experimento.\n\n¡Gracias por participar!',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('welcome.started', globalClock.getTime())
    key_resp_welcome.keys = []
    key_resp_welcome.rt = []
    _key_resp_welcome_allKeys = []
    # keep track of which components have finished
    welcomeComponents = [text_welcome, key_resp_welcome]
    for thisComponent in welcomeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_welcome* updates
        
        # if text_welcome is starting this frame...
        if text_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_welcome.frameNStart = frameN  # exact frame index
            text_welcome.tStart = t  # local t and not account for scr refresh
            text_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_welcome.started')
            # update status
            text_welcome.status = STARTED
            text_welcome.setAutoDraw(True)
        
        # if text_welcome is active this frame...
        if text_welcome.status == STARTED:
            # update params
            pass
        
        # *key_resp_welcome* updates
        waitOnFlip = False
        
        # if key_resp_welcome is starting this frame...
        if key_resp_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_welcome.frameNStart = frameN  # exact frame index
            key_resp_welcome.tStart = t  # local t and not account for scr refresh
            key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_welcome.started')
            # update status
            key_resp_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_welcome_allKeys.extend(theseKeys)
            if len(_key_resp_welcome_allKeys):
                key_resp_welcome.keys = _key_resp_welcome_allKeys[-1].name  # just the last key pressed
                key_resp_welcome.rt = _key_resp_welcome_allKeys[-1].rt
                key_resp_welcome.duration = _key_resp_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('welcome.stopped', globalClock.getTime())
    # check responses
    if key_resp_welcome.keys in ['', [], None]:  # No response was made
        key_resp_welcome.keys = None
    thisExp.addData('key_resp_welcome.keys',key_resp_welcome.keys)
    if key_resp_welcome.keys != None:  # we had a response
        thisExp.addData('key_resp_welcome.rt', key_resp_welcome.rt)
        thisExp.addData('key_resp_welcome.duration', key_resp_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_cal" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_cal.started', globalClock.getTime())
    key_resp_intro_cal.keys = []
    key_resp_intro_cal.rt = []
    _key_resp_intro_cal_allKeys = []
    # Run 'Begin Routine' code from code_intro_cal
    win.mouseVisible = False
    # keep track of which components have finished
    intro_calComponents = [text_intro_cal, key_resp_intro_cal]
    for thisComponent in intro_calComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_cal" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_cal* updates
        
        # if text_intro_cal is starting this frame...
        if text_intro_cal.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_cal.frameNStart = frameN  # exact frame index
            text_intro_cal.tStart = t  # local t and not account for scr refresh
            text_intro_cal.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_cal, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_cal.started')
            # update status
            text_intro_cal.status = STARTED
            text_intro_cal.setAutoDraw(True)
        
        # if text_intro_cal is active this frame...
        if text_intro_cal.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_cal* updates
        waitOnFlip = False
        
        # if key_resp_intro_cal is starting this frame...
        if key_resp_intro_cal.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_cal.frameNStart = frameN  # exact frame index
            key_resp_intro_cal.tStart = t  # local t and not account for scr refresh
            key_resp_intro_cal.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_cal, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_cal.started')
            # update status
            key_resp_intro_cal.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_cal.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_cal.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_cal.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_cal.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_cal_allKeys.extend(theseKeys)
            if len(_key_resp_intro_cal_allKeys):
                key_resp_intro_cal.keys = _key_resp_intro_cal_allKeys[-1].name  # just the last key pressed
                key_resp_intro_cal.rt = _key_resp_intro_cal_allKeys[-1].rt
                key_resp_intro_cal.duration = _key_resp_intro_cal_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_calComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_cal" ---
    for thisComponent in intro_calComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_cal.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_cal.keys in ['', [], None]:  # No response was made
        key_resp_intro_cal.keys = None
    thisExp.addData('key_resp_intro_cal.keys',key_resp_intro_cal.keys)
    if key_resp_intro_cal.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_cal.rt', key_resp_intro_cal.rt)
        thisExp.addData('key_resp_intro_cal.duration', key_resp_intro_cal.duration)
    thisExp.nextEntry()
    # the Routine "intro_cal" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "fix" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('fix.started', globalClock.getTime())
    # keep track of which components have finished
    fixComponents = [cross, text_nothing]
    for thisComponent in fixComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "fix" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *cross* updates
        
        # if cross is starting this frame...
        if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            cross.frameNStart = frameN  # exact frame index
            cross.tStart = t  # local t and not account for scr refresh
            cross.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'cross.started')
            # update status
            cross.status = STARTED
            cross.setAutoDraw(True)
        
        # if cross is active this frame...
        if cross.status == STARTED:
            # update params
            pass
        
        # if cross is stopping this frame...
        if cross.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > cross.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                cross.tStop = t  # not accounting for scr refresh
                cross.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.stopped')
                # update status
                cross.status = FINISHED
                cross.setAutoDraw(False)
        
        # *text_nothing* updates
        
        # if text_nothing is starting this frame...
        if text_nothing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_nothing.frameNStart = frameN  # exact frame index
            text_nothing.tStart = t  # local t and not account for scr refresh
            text_nothing.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_nothing, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_nothing.started')
            # update status
            text_nothing.status = STARTED
            text_nothing.setAutoDraw(True)
        
        # if text_nothing is active this frame...
        if text_nothing.status == STARTED:
            # update params
            pass
        
        # if text_nothing is stopping this frame...
        if text_nothing.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_nothing.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_nothing.tStop = t  # not accounting for scr refresh
                text_nothing.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_nothing.stopped')
                # update status
                text_nothing.status = FINISHED
                text_nothing.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in fixComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "fix" ---
    for thisComponent in fixComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('fix.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # --- Prepare to start Routine "calibration" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('calibration.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_cal
    log.info("Start calibration")
    calibration = tr.ScreenBasedCalibration(et)
    
    calibration.enter_calibration_mode()
    points_to_calibrate = [circle_cal_1.pos, circle_cal_2.pos, circle_cal_3.pos, circle_cal_4.pos, circle_cal_5.pos]
    
    framerate = 60
    
    # keep track of which components have finished
    calibrationComponents = [circle_cal_1, circle_cal_2, circle_cal_3, circle_cal_4, circle_cal_5]
    for thisComponent in calibrationComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 15.3:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *circle_cal_1* updates
        
        # if circle_cal_1 is starting this frame...
        if circle_cal_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            circle_cal_1.frameNStart = frameN  # exact frame index
            circle_cal_1.tStart = t  # local t and not account for scr refresh
            circle_cal_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_cal_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_cal_1.started')
            # update status
            circle_cal_1.status = STARTED
            circle_cal_1.setAutoDraw(True)
        
        # if circle_cal_1 is active this frame...
        if circle_cal_1.status == STARTED:
            # update params
            pass
        
        # if circle_cal_1 is stopping this frame...
        if circle_cal_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > circle_cal_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                circle_cal_1.tStop = t  # not accounting for scr refresh
                circle_cal_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_cal_1.stopped')
                # update status
                circle_cal_1.status = FINISHED
                circle_cal_1.setAutoDraw(False)
        
        # *circle_cal_2* updates
        
        # if circle_cal_2 is starting this frame...
        if circle_cal_2.status == NOT_STARTED and tThisFlip >= 3.1-frameTolerance:
            # keep track of start time/frame for later
            circle_cal_2.frameNStart = frameN  # exact frame index
            circle_cal_2.tStart = t  # local t and not account for scr refresh
            circle_cal_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_cal_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_cal_2.started')
            # update status
            circle_cal_2.status = STARTED
            circle_cal_2.setAutoDraw(True)
        
        # if circle_cal_2 is active this frame...
        if circle_cal_2.status == STARTED:
            # update params
            pass
        
        # if circle_cal_2 is stopping this frame...
        if circle_cal_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > circle_cal_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                circle_cal_2.tStop = t  # not accounting for scr refresh
                circle_cal_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_cal_2.stopped')
                # update status
                circle_cal_2.status = FINISHED
                circle_cal_2.setAutoDraw(False)
        
        # *circle_cal_3* updates
        
        # if circle_cal_3 is starting this frame...
        if circle_cal_3.status == NOT_STARTED and tThisFlip >= 6.2-frameTolerance:
            # keep track of start time/frame for later
            circle_cal_3.frameNStart = frameN  # exact frame index
            circle_cal_3.tStart = t  # local t and not account for scr refresh
            circle_cal_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_cal_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_cal_3.started')
            # update status
            circle_cal_3.status = STARTED
            circle_cal_3.setAutoDraw(True)
        
        # if circle_cal_3 is active this frame...
        if circle_cal_3.status == STARTED:
            # update params
            pass
        
        # if circle_cal_3 is stopping this frame...
        if circle_cal_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > circle_cal_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                circle_cal_3.tStop = t  # not accounting for scr refresh
                circle_cal_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_cal_3.stopped')
                # update status
                circle_cal_3.status = FINISHED
                circle_cal_3.setAutoDraw(False)
        
        # *circle_cal_4* updates
        
        # if circle_cal_4 is starting this frame...
        if circle_cal_4.status == NOT_STARTED and tThisFlip >= 9.3-frameTolerance:
            # keep track of start time/frame for later
            circle_cal_4.frameNStart = frameN  # exact frame index
            circle_cal_4.tStart = t  # local t and not account for scr refresh
            circle_cal_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_cal_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_cal_4.started')
            # update status
            circle_cal_4.status = STARTED
            circle_cal_4.setAutoDraw(True)
        
        # if circle_cal_4 is active this frame...
        if circle_cal_4.status == STARTED:
            # update params
            pass
        
        # if circle_cal_4 is stopping this frame...
        if circle_cal_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > circle_cal_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                circle_cal_4.tStop = t  # not accounting for scr refresh
                circle_cal_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_cal_4.stopped')
                # update status
                circle_cal_4.status = FINISHED
                circle_cal_4.setAutoDraw(False)
        
        # *circle_cal_5* updates
        
        # if circle_cal_5 is starting this frame...
        if circle_cal_5.status == NOT_STARTED and tThisFlip >= 12.3-frameTolerance:
            # keep track of start time/frame for later
            circle_cal_5.frameNStart = frameN  # exact frame index
            circle_cal_5.tStart = t  # local t and not account for scr refresh
            circle_cal_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(circle_cal_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'circle_cal_5.started')
            # update status
            circle_cal_5.status = STARTED
            circle_cal_5.setAutoDraw(True)
        
        # if circle_cal_5 is active this frame...
        if circle_cal_5.status == STARTED:
            # update params
            pass
        
        # if circle_cal_5 is stopping this frame...
        if circle_cal_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > circle_cal_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                circle_cal_5.tStop = t  # not accounting for scr refresh
                circle_cal_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'circle_cal_5.stopped')
                # update status
                circle_cal_5.status = FINISHED
                circle_cal_5.setAutoDraw(False)
        # Run 'Each Frame' code from code_cal
        if t == 0 * framerate:
            status = calibration.collect_data(points_to_calibrate[0][0], points_to_calibrate[0][1])
        elif t == 3.1 * framerate:
            status = calibration.collect_data(points_to_calibrate[1][0], points_to_calibrate[1][1])
        elif t == 6.2 * framerate:
            status = calibration.collect_data(points_to_calibrate[2][0], points_to_calibrate[2][1])
        elif t == 9.3 * framerate:
            status = calibration.collect_data(points_to_calibrate[3][0], points_to_calibrate[3][1])
        elif t == 12.3 * framerate:
            status = calibration.collect_data(points_to_calibrate[4][0], points_to_calibrate[4][1])
            
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in calibrationComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration" ---
    for thisComponent in calibrationComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('calibration.stopped', globalClock.getTime())
    # Run 'End Routine' code from code_cal
    calibration_result = calibration.compute_and_apply()
    log.info(f"Compute and apply returned {calibration_result.status} and collected at {len(calibration_result.calibration_points)} points.")
    calibration.leave_calibration_mode()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-15.300000)
    
    # --- Prepare to start Routine "intro" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro.started', globalClock.getTime())
    key_resp_intro.keys = []
    key_resp_intro.rt = []
    _key_resp_intro_allKeys = []
    # keep track of which components have finished
    introComponents = [text_intro, key_resp_intro]
    for thisComponent in introComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro* updates
        
        # if text_intro is starting this frame...
        if text_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro.frameNStart = frameN  # exact frame index
            text_intro.tStart = t  # local t and not account for scr refresh
            text_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro.started')
            # update status
            text_intro.status = STARTED
            text_intro.setAutoDraw(True)
        
        # if text_intro is active this frame...
        if text_intro.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro* updates
        waitOnFlip = False
        
        # if key_resp_intro is starting this frame...
        if key_resp_intro.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro.frameNStart = frameN  # exact frame index
            key_resp_intro.tStart = t  # local t and not account for scr refresh
            key_resp_intro.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro.started')
            # update status
            key_resp_intro.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_allKeys.extend(theseKeys)
            if len(_key_resp_intro_allKeys):
                key_resp_intro.keys = _key_resp_intro_allKeys[-1].name  # just the last key pressed
                key_resp_intro.rt = _key_resp_intro_allKeys[-1].rt
                key_resp_intro.duration = _key_resp_intro_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in introComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro" ---
    for thisComponent in introComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro.keys in ['', [], None]:  # No response was made
        key_resp_intro.keys = None
    thisExp.addData('key_resp_intro.keys',key_resp_intro.keys)
    if key_resp_intro.keys != None:  # we had a response
        thisExp.addData('key_resp_intro.rt', key_resp_intro.rt)
        thisExp.addData('key_resp_intro.duration', key_resp_intro.duration)
    thisExp.nextEntry()
    # the Routine "intro" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_ex1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_ex1.started', globalClock.getTime())
    key_resp_intro_ex1.keys = []
    key_resp_intro_ex1.rt = []
    _key_resp_intro_ex1_allKeys = []
    # keep track of which components have finished
    intro_ex1Components = [text_intro_ex1, key_resp_intro_ex1]
    for thisComponent in intro_ex1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_ex1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_ex1* updates
        
        # if text_intro_ex1 is starting this frame...
        if text_intro_ex1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_ex1.frameNStart = frameN  # exact frame index
            text_intro_ex1.tStart = t  # local t and not account for scr refresh
            text_intro_ex1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_ex1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_ex1.started')
            # update status
            text_intro_ex1.status = STARTED
            text_intro_ex1.setAutoDraw(True)
        
        # if text_intro_ex1 is active this frame...
        if text_intro_ex1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_ex1* updates
        waitOnFlip = False
        
        # if key_resp_intro_ex1 is starting this frame...
        if key_resp_intro_ex1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_ex1.frameNStart = frameN  # exact frame index
            key_resp_intro_ex1.tStart = t  # local t and not account for scr refresh
            key_resp_intro_ex1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_ex1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_ex1.started')
            # update status
            key_resp_intro_ex1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_ex1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_ex1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_ex1.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_ex1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_ex1_allKeys.extend(theseKeys)
            if len(_key_resp_intro_ex1_allKeys):
                key_resp_intro_ex1.keys = _key_resp_intro_ex1_allKeys[-1].name  # just the last key pressed
                key_resp_intro_ex1.rt = _key_resp_intro_ex1_allKeys[-1].rt
                key_resp_intro_ex1.duration = _key_resp_intro_ex1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_ex1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_ex1" ---
    for thisComponent in intro_ex1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_ex1.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_ex1.keys in ['', [], None]:  # No response was made
        key_resp_intro_ex1.keys = None
    thisExp.addData('key_resp_intro_ex1.keys',key_resp_intro_ex1.keys)
    if key_resp_intro_ex1.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_ex1.rt', key_resp_intro_ex1.rt)
        thisExp.addData('key_resp_intro_ex1.duration', key_resp_intro_ex1.duration)
    thisExp.nextEntry()
    # the Routine "intro_ex1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ex1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ex1.started', globalClock.getTime())
    key_resp_ex1.keys = []
    key_resp_ex1.rt = []
    _key_resp_ex1_allKeys = []
    # keep track of which components have finished
    ex1Components = [text_ex1_1, rect_ex1_1, text_ex1_2, rect_ex1_2, text_ex1_3, rect_ex1_3, text_ex1_4, rect_ex1_4, text_ex1_5, rect_ex1_5, text_ex1_6, rect_ex1_6, text_ex1_array, key_resp_ex1, text_ex1_cue]
    for thisComponent in ex1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ex1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 18.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ex1_1* updates
        
        # if text_ex1_1 is starting this frame...
        if text_ex1_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_1.frameNStart = frameN  # exact frame index
            text_ex1_1.tStart = t  # local t and not account for scr refresh
            text_ex1_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_1.started')
            # update status
            text_ex1_1.status = STARTED
            text_ex1_1.setAutoDraw(True)
        
        # if text_ex1_1 is active this frame...
        if text_ex1_1.status == STARTED:
            # update params
            pass
        
        # if text_ex1_1 is stopping this frame...
        if text_ex1_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_1.tStop = t  # not accounting for scr refresh
                text_ex1_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_1.stopped')
                # update status
                text_ex1_1.status = FINISHED
                text_ex1_1.setAutoDraw(False)
        
        # *rect_ex1_1* updates
        
        # if rect_ex1_1 is starting this frame...
        if rect_ex1_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_1.frameNStart = frameN  # exact frame index
            rect_ex1_1.tStart = t  # local t and not account for scr refresh
            rect_ex1_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_1.started')
            # update status
            rect_ex1_1.status = STARTED
            rect_ex1_1.setAutoDraw(True)
        
        # if rect_ex1_1 is active this frame...
        if rect_ex1_1.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_1 is stopping this frame...
        if rect_ex1_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_1.tStop = t  # not accounting for scr refresh
                rect_ex1_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_1.stopped')
                # update status
                rect_ex1_1.status = FINISHED
                rect_ex1_1.setAutoDraw(False)
        
        # *text_ex1_2* updates
        
        # if text_ex1_2 is starting this frame...
        if text_ex1_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_2.frameNStart = frameN  # exact frame index
            text_ex1_2.tStart = t  # local t and not account for scr refresh
            text_ex1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_2.started')
            # update status
            text_ex1_2.status = STARTED
            text_ex1_2.setAutoDraw(True)
        
        # if text_ex1_2 is active this frame...
        if text_ex1_2.status == STARTED:
            # update params
            pass
        
        # if text_ex1_2 is stopping this frame...
        if text_ex1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_2.tStop = t  # not accounting for scr refresh
                text_ex1_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_2.stopped')
                # update status
                text_ex1_2.status = FINISHED
                text_ex1_2.setAutoDraw(False)
        
        # *rect_ex1_2* updates
        
        # if rect_ex1_2 is starting this frame...
        if rect_ex1_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_2.frameNStart = frameN  # exact frame index
            rect_ex1_2.tStart = t  # local t and not account for scr refresh
            rect_ex1_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_2.started')
            # update status
            rect_ex1_2.status = STARTED
            rect_ex1_2.setAutoDraw(True)
        
        # if rect_ex1_2 is active this frame...
        if rect_ex1_2.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_2 is stopping this frame...
        if rect_ex1_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_2.tStop = t  # not accounting for scr refresh
                rect_ex1_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_2.stopped')
                # update status
                rect_ex1_2.status = FINISHED
                rect_ex1_2.setAutoDraw(False)
        
        # *text_ex1_3* updates
        
        # if text_ex1_3 is starting this frame...
        if text_ex1_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_3.frameNStart = frameN  # exact frame index
            text_ex1_3.tStart = t  # local t and not account for scr refresh
            text_ex1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_3.started')
            # update status
            text_ex1_3.status = STARTED
            text_ex1_3.setAutoDraw(True)
        
        # if text_ex1_3 is active this frame...
        if text_ex1_3.status == STARTED:
            # update params
            pass
        
        # if text_ex1_3 is stopping this frame...
        if text_ex1_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_3.tStop = t  # not accounting for scr refresh
                text_ex1_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_3.stopped')
                # update status
                text_ex1_3.status = FINISHED
                text_ex1_3.setAutoDraw(False)
        
        # *rect_ex1_3* updates
        
        # if rect_ex1_3 is starting this frame...
        if rect_ex1_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_3.frameNStart = frameN  # exact frame index
            rect_ex1_3.tStart = t  # local t and not account for scr refresh
            rect_ex1_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_3.started')
            # update status
            rect_ex1_3.status = STARTED
            rect_ex1_3.setAutoDraw(True)
        
        # if rect_ex1_3 is active this frame...
        if rect_ex1_3.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_3 is stopping this frame...
        if rect_ex1_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_3.tStop = t  # not accounting for scr refresh
                rect_ex1_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_3.stopped')
                # update status
                rect_ex1_3.status = FINISHED
                rect_ex1_3.setAutoDraw(False)
        
        # *text_ex1_4* updates
        
        # if text_ex1_4 is starting this frame...
        if text_ex1_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_4.frameNStart = frameN  # exact frame index
            text_ex1_4.tStart = t  # local t and not account for scr refresh
            text_ex1_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_4.started')
            # update status
            text_ex1_4.status = STARTED
            text_ex1_4.setAutoDraw(True)
        
        # if text_ex1_4 is active this frame...
        if text_ex1_4.status == STARTED:
            # update params
            pass
        
        # if text_ex1_4 is stopping this frame...
        if text_ex1_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_4.tStop = t  # not accounting for scr refresh
                text_ex1_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_4.stopped')
                # update status
                text_ex1_4.status = FINISHED
                text_ex1_4.setAutoDraw(False)
        
        # *rect_ex1_4* updates
        
        # if rect_ex1_4 is starting this frame...
        if rect_ex1_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_4.frameNStart = frameN  # exact frame index
            rect_ex1_4.tStart = t  # local t and not account for scr refresh
            rect_ex1_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_4.started')
            # update status
            rect_ex1_4.status = STARTED
            rect_ex1_4.setAutoDraw(True)
        
        # if rect_ex1_4 is active this frame...
        if rect_ex1_4.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_4 is stopping this frame...
        if rect_ex1_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_4.tStop = t  # not accounting for scr refresh
                rect_ex1_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_4.stopped')
                # update status
                rect_ex1_4.status = FINISHED
                rect_ex1_4.setAutoDraw(False)
        
        # *text_ex1_5* updates
        
        # if text_ex1_5 is starting this frame...
        if text_ex1_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_5.frameNStart = frameN  # exact frame index
            text_ex1_5.tStart = t  # local t and not account for scr refresh
            text_ex1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_5.started')
            # update status
            text_ex1_5.status = STARTED
            text_ex1_5.setAutoDraw(True)
        
        # if text_ex1_5 is active this frame...
        if text_ex1_5.status == STARTED:
            # update params
            pass
        
        # if text_ex1_5 is stopping this frame...
        if text_ex1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_5.tStop = t  # not accounting for scr refresh
                text_ex1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_5.stopped')
                # update status
                text_ex1_5.status = FINISHED
                text_ex1_5.setAutoDraw(False)
        
        # *rect_ex1_5* updates
        
        # if rect_ex1_5 is starting this frame...
        if rect_ex1_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_5.frameNStart = frameN  # exact frame index
            rect_ex1_5.tStart = t  # local t and not account for scr refresh
            rect_ex1_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_5.started')
            # update status
            rect_ex1_5.status = STARTED
            rect_ex1_5.setAutoDraw(True)
        
        # if rect_ex1_5 is active this frame...
        if rect_ex1_5.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_5 is stopping this frame...
        if rect_ex1_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_5.tStop = t  # not accounting for scr refresh
                rect_ex1_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_5.stopped')
                # update status
                rect_ex1_5.status = FINISHED
                rect_ex1_5.setAutoDraw(False)
        
        # *text_ex1_6* updates
        
        # if text_ex1_6 is starting this frame...
        if text_ex1_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_6.frameNStart = frameN  # exact frame index
            text_ex1_6.tStart = t  # local t and not account for scr refresh
            text_ex1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_6.started')
            # update status
            text_ex1_6.status = STARTED
            text_ex1_6.setAutoDraw(True)
        
        # if text_ex1_6 is active this frame...
        if text_ex1_6.status == STARTED:
            # update params
            pass
        
        # if text_ex1_6 is stopping this frame...
        if text_ex1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_6.tStop = t  # not accounting for scr refresh
                text_ex1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_6.stopped')
                # update status
                text_ex1_6.status = FINISHED
                text_ex1_6.setAutoDraw(False)
        
        # *rect_ex1_6* updates
        
        # if rect_ex1_6 is starting this frame...
        if rect_ex1_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            rect_ex1_6.frameNStart = frameN  # exact frame index
            rect_ex1_6.tStart = t  # local t and not account for scr refresh
            rect_ex1_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex1_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex1_6.started')
            # update status
            rect_ex1_6.status = STARTED
            rect_ex1_6.setAutoDraw(True)
        
        # if rect_ex1_6 is active this frame...
        if rect_ex1_6.status == STARTED:
            # update params
            pass
        
        # if rect_ex1_6 is stopping this frame...
        if rect_ex1_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex1_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex1_6.tStop = t  # not accounting for scr refresh
                rect_ex1_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex1_6.stopped')
                # update status
                rect_ex1_6.status = FINISHED
                rect_ex1_6.setAutoDraw(False)
        
        # *text_ex1_array* updates
        
        # if text_ex1_array is starting this frame...
        if text_ex1_array.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_array.frameNStart = frameN  # exact frame index
            text_ex1_array.tStart = t  # local t and not account for scr refresh
            text_ex1_array.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_array, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_array.started')
            # update status
            text_ex1_array.status = STARTED
            text_ex1_array.setAutoDraw(True)
        
        # if text_ex1_array is active this frame...
        if text_ex1_array.status == STARTED:
            # update params
            pass
        
        # if text_ex1_array is stopping this frame...
        if text_ex1_array.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_array.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_array.tStop = t  # not accounting for scr refresh
                text_ex1_array.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_array.stopped')
                # update status
                text_ex1_array.status = FINISHED
                text_ex1_array.setAutoDraw(False)
        
        # *key_resp_ex1* updates
        waitOnFlip = False
        
        # if key_resp_ex1 is starting this frame...
        if key_resp_ex1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ex1.frameNStart = frameN  # exact frame index
            key_resp_ex1.tStart = t  # local t and not account for scr refresh
            key_resp_ex1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ex1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ex1.started')
            # update status
            key_resp_ex1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ex1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ex1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_ex1 is stopping this frame...
        if key_resp_ex1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_ex1.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_ex1.tStop = t  # not accounting for scr refresh
                key_resp_ex1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_ex1.stopped')
                # update status
                key_resp_ex1.status = FINISHED
                key_resp_ex1.status = FINISHED
        if key_resp_ex1.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ex1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ex1_allKeys.extend(theseKeys)
            if len(_key_resp_ex1_allKeys):
                key_resp_ex1.keys = _key_resp_ex1_allKeys[-1].name  # just the last key pressed
                key_resp_ex1.rt = _key_resp_ex1_allKeys[-1].rt
                key_resp_ex1.duration = _key_resp_ex1_allKeys[-1].duration
        
        # *text_ex1_cue* updates
        
        # if text_ex1_cue is starting this frame...
        if text_ex1_cue.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex1_cue.frameNStart = frameN  # exact frame index
            text_ex1_cue.tStart = t  # local t and not account for scr refresh
            text_ex1_cue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex1_cue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex1_cue.started')
            # update status
            text_ex1_cue.status = STARTED
            text_ex1_cue.setAutoDraw(True)
        
        # if text_ex1_cue is active this frame...
        if text_ex1_cue.status == STARTED:
            # update params
            pass
        
        # if text_ex1_cue is stopping this frame...
        if text_ex1_cue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex1_cue.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex1_cue.tStop = t  # not accounting for scr refresh
                text_ex1_cue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex1_cue.stopped')
                # update status
                text_ex1_cue.status = FINISHED
                text_ex1_cue.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ex1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ex1" ---
    for thisComponent in ex1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ex1.stopped', globalClock.getTime())
    # check responses
    if key_resp_ex1.keys in ['', [], None]:  # No response was made
        key_resp_ex1.keys = None
    thisExp.addData('key_resp_ex1.keys',key_resp_ex1.keys)
    if key_resp_ex1.keys != None:  # we had a response
        thisExp.addData('key_resp_ex1.rt', key_resp_ex1.rt)
        thisExp.addData('key_resp_ex1.duration', key_resp_ex1.duration)
    thisExp.nextEntry()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-18.000000)
    
    # --- Prepare to start Routine "intro_ex2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_ex2.started', globalClock.getTime())
    key_resp_intro_ex2.keys = []
    key_resp_intro_ex2.rt = []
    _key_resp_intro_ex2_allKeys = []
    # keep track of which components have finished
    intro_ex2Components = [text_intro_ex2, key_resp_intro_ex2]
    for thisComponent in intro_ex2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_ex2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_ex2* updates
        
        # if text_intro_ex2 is starting this frame...
        if text_intro_ex2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_ex2.frameNStart = frameN  # exact frame index
            text_intro_ex2.tStart = t  # local t and not account for scr refresh
            text_intro_ex2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_ex2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_ex2.started')
            # update status
            text_intro_ex2.status = STARTED
            text_intro_ex2.setAutoDraw(True)
        
        # if text_intro_ex2 is active this frame...
        if text_intro_ex2.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_ex2* updates
        waitOnFlip = False
        
        # if key_resp_intro_ex2 is starting this frame...
        if key_resp_intro_ex2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_ex2.frameNStart = frameN  # exact frame index
            key_resp_intro_ex2.tStart = t  # local t and not account for scr refresh
            key_resp_intro_ex2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_ex2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_ex2.started')
            # update status
            key_resp_intro_ex2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_ex2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_ex2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_ex2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_ex2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_ex2_allKeys.extend(theseKeys)
            if len(_key_resp_intro_ex2_allKeys):
                key_resp_intro_ex2.keys = _key_resp_intro_ex2_allKeys[-1].name  # just the last key pressed
                key_resp_intro_ex2.rt = _key_resp_intro_ex2_allKeys[-1].rt
                key_resp_intro_ex2.duration = _key_resp_intro_ex2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_ex2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_ex2" ---
    for thisComponent in intro_ex2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_ex2.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_ex2.keys in ['', [], None]:  # No response was made
        key_resp_intro_ex2.keys = None
    thisExp.addData('key_resp_intro_ex2.keys',key_resp_intro_ex2.keys)
    if key_resp_intro_ex2.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_ex2.rt', key_resp_intro_ex2.rt)
        thisExp.addData('key_resp_intro_ex2.duration', key_resp_intro_ex2.duration)
    thisExp.nextEntry()
    # the Routine "intro_ex2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ex2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ex2.started', globalClock.getTime())
    key_resp_ex2.keys = []
    key_resp_ex2.rt = []
    _key_resp_ex2_allKeys = []
    # keep track of which components have finished
    ex2Components = [text_ex2_1, rect_ex2_1, text_ex2_2, rect_ex2_2, text_ex2_3, rect_ex2_3, text_ex2_4, rect_ex2_4, text_ex2_5, rect_ex2_5, text_ex2_6, rect_ex2_6, text_ex2_array, key_resp_ex2, text_ex2_cue]
    for thisComponent in ex2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ex2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 18.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ex2_1* updates
        
        # if text_ex2_1 is starting this frame...
        if text_ex2_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_1.frameNStart = frameN  # exact frame index
            text_ex2_1.tStart = t  # local t and not account for scr refresh
            text_ex2_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_1.started')
            # update status
            text_ex2_1.status = STARTED
            text_ex2_1.setAutoDraw(True)
        
        # if text_ex2_1 is active this frame...
        if text_ex2_1.status == STARTED:
            # update params
            pass
        
        # if text_ex2_1 is stopping this frame...
        if text_ex2_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_1.tStop = t  # not accounting for scr refresh
                text_ex2_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_1.stopped')
                # update status
                text_ex2_1.status = FINISHED
                text_ex2_1.setAutoDraw(False)
        
        # *rect_ex2_1* updates
        
        # if rect_ex2_1 is starting this frame...
        if rect_ex2_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_1.frameNStart = frameN  # exact frame index
            rect_ex2_1.tStart = t  # local t and not account for scr refresh
            rect_ex2_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_1.started')
            # update status
            rect_ex2_1.status = STARTED
            rect_ex2_1.setAutoDraw(True)
        
        # if rect_ex2_1 is active this frame...
        if rect_ex2_1.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_1 is stopping this frame...
        if rect_ex2_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_1.tStop = t  # not accounting for scr refresh
                rect_ex2_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_1.stopped')
                # update status
                rect_ex2_1.status = FINISHED
                rect_ex2_1.setAutoDraw(False)
        
        # *text_ex2_2* updates
        
        # if text_ex2_2 is starting this frame...
        if text_ex2_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_2.frameNStart = frameN  # exact frame index
            text_ex2_2.tStart = t  # local t and not account for scr refresh
            text_ex2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_2.started')
            # update status
            text_ex2_2.status = STARTED
            text_ex2_2.setAutoDraw(True)
        
        # if text_ex2_2 is active this frame...
        if text_ex2_2.status == STARTED:
            # update params
            pass
        
        # if text_ex2_2 is stopping this frame...
        if text_ex2_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_2.tStop = t  # not accounting for scr refresh
                text_ex2_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_2.stopped')
                # update status
                text_ex2_2.status = FINISHED
                text_ex2_2.setAutoDraw(False)
        
        # *rect_ex2_2* updates
        
        # if rect_ex2_2 is starting this frame...
        if rect_ex2_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_2.frameNStart = frameN  # exact frame index
            rect_ex2_2.tStart = t  # local t and not account for scr refresh
            rect_ex2_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_2.started')
            # update status
            rect_ex2_2.status = STARTED
            rect_ex2_2.setAutoDraw(True)
        
        # if rect_ex2_2 is active this frame...
        if rect_ex2_2.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_2 is stopping this frame...
        if rect_ex2_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_2.tStop = t  # not accounting for scr refresh
                rect_ex2_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_2.stopped')
                # update status
                rect_ex2_2.status = FINISHED
                rect_ex2_2.setAutoDraw(False)
        
        # *text_ex2_3* updates
        
        # if text_ex2_3 is starting this frame...
        if text_ex2_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_3.frameNStart = frameN  # exact frame index
            text_ex2_3.tStart = t  # local t and not account for scr refresh
            text_ex2_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_3.started')
            # update status
            text_ex2_3.status = STARTED
            text_ex2_3.setAutoDraw(True)
        
        # if text_ex2_3 is active this frame...
        if text_ex2_3.status == STARTED:
            # update params
            pass
        
        # if text_ex2_3 is stopping this frame...
        if text_ex2_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_3.tStop = t  # not accounting for scr refresh
                text_ex2_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_3.stopped')
                # update status
                text_ex2_3.status = FINISHED
                text_ex2_3.setAutoDraw(False)
        
        # *rect_ex2_3* updates
        
        # if rect_ex2_3 is starting this frame...
        if rect_ex2_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_3.frameNStart = frameN  # exact frame index
            rect_ex2_3.tStart = t  # local t and not account for scr refresh
            rect_ex2_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_3.started')
            # update status
            rect_ex2_3.status = STARTED
            rect_ex2_3.setAutoDraw(True)
        
        # if rect_ex2_3 is active this frame...
        if rect_ex2_3.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_3 is stopping this frame...
        if rect_ex2_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_3.tStop = t  # not accounting for scr refresh
                rect_ex2_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_3.stopped')
                # update status
                rect_ex2_3.status = FINISHED
                rect_ex2_3.setAutoDraw(False)
        
        # *text_ex2_4* updates
        
        # if text_ex2_4 is starting this frame...
        if text_ex2_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_4.frameNStart = frameN  # exact frame index
            text_ex2_4.tStart = t  # local t and not account for scr refresh
            text_ex2_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_4.started')
            # update status
            text_ex2_4.status = STARTED
            text_ex2_4.setAutoDraw(True)
        
        # if text_ex2_4 is active this frame...
        if text_ex2_4.status == STARTED:
            # update params
            pass
        
        # if text_ex2_4 is stopping this frame...
        if text_ex2_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_4.tStop = t  # not accounting for scr refresh
                text_ex2_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_4.stopped')
                # update status
                text_ex2_4.status = FINISHED
                text_ex2_4.setAutoDraw(False)
        
        # *rect_ex2_4* updates
        
        # if rect_ex2_4 is starting this frame...
        if rect_ex2_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_4.frameNStart = frameN  # exact frame index
            rect_ex2_4.tStart = t  # local t and not account for scr refresh
            rect_ex2_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_4.started')
            # update status
            rect_ex2_4.status = STARTED
            rect_ex2_4.setAutoDraw(True)
        
        # if rect_ex2_4 is active this frame...
        if rect_ex2_4.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_4 is stopping this frame...
        if rect_ex2_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_4.tStop = t  # not accounting for scr refresh
                rect_ex2_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_4.stopped')
                # update status
                rect_ex2_4.status = FINISHED
                rect_ex2_4.setAutoDraw(False)
        
        # *text_ex2_5* updates
        
        # if text_ex2_5 is starting this frame...
        if text_ex2_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_5.frameNStart = frameN  # exact frame index
            text_ex2_5.tStart = t  # local t and not account for scr refresh
            text_ex2_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_5.started')
            # update status
            text_ex2_5.status = STARTED
            text_ex2_5.setAutoDraw(True)
        
        # if text_ex2_5 is active this frame...
        if text_ex2_5.status == STARTED:
            # update params
            pass
        
        # if text_ex2_5 is stopping this frame...
        if text_ex2_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_5.tStop = t  # not accounting for scr refresh
                text_ex2_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_5.stopped')
                # update status
                text_ex2_5.status = FINISHED
                text_ex2_5.setAutoDraw(False)
        
        # *rect_ex2_5* updates
        
        # if rect_ex2_5 is starting this frame...
        if rect_ex2_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_5.frameNStart = frameN  # exact frame index
            rect_ex2_5.tStart = t  # local t and not account for scr refresh
            rect_ex2_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_5.started')
            # update status
            rect_ex2_5.status = STARTED
            rect_ex2_5.setAutoDraw(True)
        
        # if rect_ex2_5 is active this frame...
        if rect_ex2_5.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_5 is stopping this frame...
        if rect_ex2_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_5.tStop = t  # not accounting for scr refresh
                rect_ex2_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_5.stopped')
                # update status
                rect_ex2_5.status = FINISHED
                rect_ex2_5.setAutoDraw(False)
        
        # *text_ex2_6* updates
        
        # if text_ex2_6 is starting this frame...
        if text_ex2_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_6.frameNStart = frameN  # exact frame index
            text_ex2_6.tStart = t  # local t and not account for scr refresh
            text_ex2_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_6.started')
            # update status
            text_ex2_6.status = STARTED
            text_ex2_6.setAutoDraw(True)
        
        # if text_ex2_6 is active this frame...
        if text_ex2_6.status == STARTED:
            # update params
            pass
        
        # if text_ex2_6 is stopping this frame...
        if text_ex2_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_6.tStop = t  # not accounting for scr refresh
                text_ex2_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_6.stopped')
                # update status
                text_ex2_6.status = FINISHED
                text_ex2_6.setAutoDraw(False)
        
        # *rect_ex2_6* updates
        
        # if rect_ex2_6 is starting this frame...
        if rect_ex2_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            rect_ex2_6.frameNStart = frameN  # exact frame index
            rect_ex2_6.tStart = t  # local t and not account for scr refresh
            rect_ex2_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex2_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex2_6.started')
            # update status
            rect_ex2_6.status = STARTED
            rect_ex2_6.setAutoDraw(True)
        
        # if rect_ex2_6 is active this frame...
        if rect_ex2_6.status == STARTED:
            # update params
            pass
        
        # if rect_ex2_6 is stopping this frame...
        if rect_ex2_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex2_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex2_6.tStop = t  # not accounting for scr refresh
                rect_ex2_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex2_6.stopped')
                # update status
                rect_ex2_6.status = FINISHED
                rect_ex2_6.setAutoDraw(False)
        
        # *text_ex2_array* updates
        
        # if text_ex2_array is starting this frame...
        if text_ex2_array.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_array.frameNStart = frameN  # exact frame index
            text_ex2_array.tStart = t  # local t and not account for scr refresh
            text_ex2_array.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_array, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_array.started')
            # update status
            text_ex2_array.status = STARTED
            text_ex2_array.setAutoDraw(True)
        
        # if text_ex2_array is active this frame...
        if text_ex2_array.status == STARTED:
            # update params
            pass
        
        # if text_ex2_array is stopping this frame...
        if text_ex2_array.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_array.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_array.tStop = t  # not accounting for scr refresh
                text_ex2_array.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_array.stopped')
                # update status
                text_ex2_array.status = FINISHED
                text_ex2_array.setAutoDraw(False)
        
        # *key_resp_ex2* updates
        waitOnFlip = False
        
        # if key_resp_ex2 is starting this frame...
        if key_resp_ex2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ex2.frameNStart = frameN  # exact frame index
            key_resp_ex2.tStart = t  # local t and not account for scr refresh
            key_resp_ex2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ex2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ex2.started')
            # update status
            key_resp_ex2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ex2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ex2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_ex2 is stopping this frame...
        if key_resp_ex2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_ex2.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_ex2.tStop = t  # not accounting for scr refresh
                key_resp_ex2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_ex2.stopped')
                # update status
                key_resp_ex2.status = FINISHED
                key_resp_ex2.status = FINISHED
        if key_resp_ex2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ex2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ex2_allKeys.extend(theseKeys)
            if len(_key_resp_ex2_allKeys):
                key_resp_ex2.keys = _key_resp_ex2_allKeys[-1].name  # just the last key pressed
                key_resp_ex2.rt = _key_resp_ex2_allKeys[-1].rt
                key_resp_ex2.duration = _key_resp_ex2_allKeys[-1].duration
        
        # *text_ex2_cue* updates
        
        # if text_ex2_cue is starting this frame...
        if text_ex2_cue.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex2_cue.frameNStart = frameN  # exact frame index
            text_ex2_cue.tStart = t  # local t and not account for scr refresh
            text_ex2_cue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex2_cue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex2_cue.started')
            # update status
            text_ex2_cue.status = STARTED
            text_ex2_cue.setAutoDraw(True)
        
        # if text_ex2_cue is active this frame...
        if text_ex2_cue.status == STARTED:
            # update params
            pass
        
        # if text_ex2_cue is stopping this frame...
        if text_ex2_cue.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex2_cue.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex2_cue.tStop = t  # not accounting for scr refresh
                text_ex2_cue.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex2_cue.stopped')
                # update status
                text_ex2_cue.status = FINISHED
                text_ex2_cue.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ex2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ex2" ---
    for thisComponent in ex2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ex2.stopped', globalClock.getTime())
    # check responses
    if key_resp_ex2.keys in ['', [], None]:  # No response was made
        key_resp_ex2.keys = None
    thisExp.addData('key_resp_ex2.keys',key_resp_ex2.keys)
    if key_resp_ex2.keys != None:  # we had a response
        thisExp.addData('key_resp_ex2.rt', key_resp_ex2.rt)
        thisExp.addData('key_resp_ex2.duration', key_resp_ex2.duration)
    thisExp.nextEntry()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-18.000000)
    
    # --- Prepare to start Routine "intro_ex3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_ex3.started', globalClock.getTime())
    key_resp_intro_ex3.keys = []
    key_resp_intro_ex3.rt = []
    _key_resp_intro_ex3_allKeys = []
    # keep track of which components have finished
    intro_ex3Components = [text_intro_ex3, key_resp_intro_ex3]
    for thisComponent in intro_ex3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_ex3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_ex3* updates
        
        # if text_intro_ex3 is starting this frame...
        if text_intro_ex3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_ex3.frameNStart = frameN  # exact frame index
            text_intro_ex3.tStart = t  # local t and not account for scr refresh
            text_intro_ex3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_ex3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_ex3.started')
            # update status
            text_intro_ex3.status = STARTED
            text_intro_ex3.setAutoDraw(True)
        
        # if text_intro_ex3 is active this frame...
        if text_intro_ex3.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_ex3* updates
        waitOnFlip = False
        
        # if key_resp_intro_ex3 is starting this frame...
        if key_resp_intro_ex3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_ex3.frameNStart = frameN  # exact frame index
            key_resp_intro_ex3.tStart = t  # local t and not account for scr refresh
            key_resp_intro_ex3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_ex3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_ex3.started')
            # update status
            key_resp_intro_ex3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_ex3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_ex3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_ex3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_ex3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_ex3_allKeys.extend(theseKeys)
            if len(_key_resp_intro_ex3_allKeys):
                key_resp_intro_ex3.keys = _key_resp_intro_ex3_allKeys[-1].name  # just the last key pressed
                key_resp_intro_ex3.rt = _key_resp_intro_ex3_allKeys[-1].rt
                key_resp_intro_ex3.duration = _key_resp_intro_ex3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_ex3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_ex3" ---
    for thisComponent in intro_ex3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_ex3.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_ex3.keys in ['', [], None]:  # No response was made
        key_resp_intro_ex3.keys = None
    thisExp.addData('key_resp_intro_ex3.keys',key_resp_intro_ex3.keys)
    if key_resp_intro_ex3.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_ex3.rt', key_resp_intro_ex3.rt)
        thisExp.addData('key_resp_intro_ex3.duration', key_resp_intro_ex3.duration)
    thisExp.nextEntry()
    # the Routine "intro_ex3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "ex3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('ex3.started', globalClock.getTime())
    key_resp_ex3.keys = []
    key_resp_ex3.rt = []
    _key_resp_ex3_allKeys = []
    # keep track of which components have finished
    ex3Components = [text_ex3_1, rect_ex3_1, text_ex3_2, rect_ex3_2, text_ex3_3, rect_ex3_3, text_ex3_4, rect_ex3_4, text_ex3_5, rect_ex3_5, text_ex3_6, rect_ex3_6, text_ex3_array, key_resp_ex3, text_ex3_cue2, text_ex3_cue1]
    for thisComponent in ex3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ex3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 18.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_ex3_1* updates
        
        # if text_ex3_1 is starting this frame...
        if text_ex3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_1.frameNStart = frameN  # exact frame index
            text_ex3_1.tStart = t  # local t and not account for scr refresh
            text_ex3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_1.started')
            # update status
            text_ex3_1.status = STARTED
            text_ex3_1.setAutoDraw(True)
        
        # if text_ex3_1 is active this frame...
        if text_ex3_1.status == STARTED:
            # update params
            pass
        
        # if text_ex3_1 is stopping this frame...
        if text_ex3_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_1.tStop = t  # not accounting for scr refresh
                text_ex3_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_1.stopped')
                # update status
                text_ex3_1.status = FINISHED
                text_ex3_1.setAutoDraw(False)
        
        # *rect_ex3_1* updates
        
        # if rect_ex3_1 is starting this frame...
        if rect_ex3_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_1.frameNStart = frameN  # exact frame index
            rect_ex3_1.tStart = t  # local t and not account for scr refresh
            rect_ex3_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_1.started')
            # update status
            rect_ex3_1.status = STARTED
            rect_ex3_1.setAutoDraw(True)
        
        # if rect_ex3_1 is active this frame...
        if rect_ex3_1.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_1 is stopping this frame...
        if rect_ex3_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_1.tStop = t  # not accounting for scr refresh
                rect_ex3_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_1.stopped')
                # update status
                rect_ex3_1.status = FINISHED
                rect_ex3_1.setAutoDraw(False)
        
        # *text_ex3_2* updates
        
        # if text_ex3_2 is starting this frame...
        if text_ex3_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_2.frameNStart = frameN  # exact frame index
            text_ex3_2.tStart = t  # local t and not account for scr refresh
            text_ex3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_2.started')
            # update status
            text_ex3_2.status = STARTED
            text_ex3_2.setAutoDraw(True)
        
        # if text_ex3_2 is active this frame...
        if text_ex3_2.status == STARTED:
            # update params
            pass
        
        # if text_ex3_2 is stopping this frame...
        if text_ex3_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_2.tStop = t  # not accounting for scr refresh
                text_ex3_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_2.stopped')
                # update status
                text_ex3_2.status = FINISHED
                text_ex3_2.setAutoDraw(False)
        
        # *rect_ex3_2* updates
        
        # if rect_ex3_2 is starting this frame...
        if rect_ex3_2.status == NOT_STARTED and tThisFlip >= 3-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_2.frameNStart = frameN  # exact frame index
            rect_ex3_2.tStart = t  # local t and not account for scr refresh
            rect_ex3_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_2.started')
            # update status
            rect_ex3_2.status = STARTED
            rect_ex3_2.setAutoDraw(True)
        
        # if rect_ex3_2 is active this frame...
        if rect_ex3_2.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_2 is stopping this frame...
        if rect_ex3_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_2.tStop = t  # not accounting for scr refresh
                rect_ex3_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_2.stopped')
                # update status
                rect_ex3_2.status = FINISHED
                rect_ex3_2.setAutoDraw(False)
        
        # *text_ex3_3* updates
        
        # if text_ex3_3 is starting this frame...
        if text_ex3_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_3.frameNStart = frameN  # exact frame index
            text_ex3_3.tStart = t  # local t and not account for scr refresh
            text_ex3_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_3.started')
            # update status
            text_ex3_3.status = STARTED
            text_ex3_3.setAutoDraw(True)
        
        # if text_ex3_3 is active this frame...
        if text_ex3_3.status == STARTED:
            # update params
            pass
        
        # if text_ex3_3 is stopping this frame...
        if text_ex3_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_3.tStop = t  # not accounting for scr refresh
                text_ex3_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_3.stopped')
                # update status
                text_ex3_3.status = FINISHED
                text_ex3_3.setAutoDraw(False)
        
        # *rect_ex3_3* updates
        
        # if rect_ex3_3 is starting this frame...
        if rect_ex3_3.status == NOT_STARTED and tThisFlip >= 6-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_3.frameNStart = frameN  # exact frame index
            rect_ex3_3.tStart = t  # local t and not account for scr refresh
            rect_ex3_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_3.started')
            # update status
            rect_ex3_3.status = STARTED
            rect_ex3_3.setAutoDraw(True)
        
        # if rect_ex3_3 is active this frame...
        if rect_ex3_3.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_3 is stopping this frame...
        if rect_ex3_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_3.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_3.tStop = t  # not accounting for scr refresh
                rect_ex3_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_3.stopped')
                # update status
                rect_ex3_3.status = FINISHED
                rect_ex3_3.setAutoDraw(False)
        
        # *text_ex3_4* updates
        
        # if text_ex3_4 is starting this frame...
        if text_ex3_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_4.frameNStart = frameN  # exact frame index
            text_ex3_4.tStart = t  # local t and not account for scr refresh
            text_ex3_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_4.started')
            # update status
            text_ex3_4.status = STARTED
            text_ex3_4.setAutoDraw(True)
        
        # if text_ex3_4 is active this frame...
        if text_ex3_4.status == STARTED:
            # update params
            pass
        
        # if text_ex3_4 is stopping this frame...
        if text_ex3_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_4.tStop = t  # not accounting for scr refresh
                text_ex3_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_4.stopped')
                # update status
                text_ex3_4.status = FINISHED
                text_ex3_4.setAutoDraw(False)
        
        # *rect_ex3_4* updates
        
        # if rect_ex3_4 is starting this frame...
        if rect_ex3_4.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_4.frameNStart = frameN  # exact frame index
            rect_ex3_4.tStart = t  # local t and not account for scr refresh
            rect_ex3_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_4.started')
            # update status
            rect_ex3_4.status = STARTED
            rect_ex3_4.setAutoDraw(True)
        
        # if rect_ex3_4 is active this frame...
        if rect_ex3_4.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_4 is stopping this frame...
        if rect_ex3_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_4.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_4.tStop = t  # not accounting for scr refresh
                rect_ex3_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_4.stopped')
                # update status
                rect_ex3_4.status = FINISHED
                rect_ex3_4.setAutoDraw(False)
        
        # *text_ex3_5* updates
        
        # if text_ex3_5 is starting this frame...
        if text_ex3_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_5.frameNStart = frameN  # exact frame index
            text_ex3_5.tStart = t  # local t and not account for scr refresh
            text_ex3_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_5.started')
            # update status
            text_ex3_5.status = STARTED
            text_ex3_5.setAutoDraw(True)
        
        # if text_ex3_5 is active this frame...
        if text_ex3_5.status == STARTED:
            # update params
            pass
        
        # if text_ex3_5 is stopping this frame...
        if text_ex3_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_5.tStop = t  # not accounting for scr refresh
                text_ex3_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_5.stopped')
                # update status
                text_ex3_5.status = FINISHED
                text_ex3_5.setAutoDraw(False)
        
        # *rect_ex3_5* updates
        
        # if rect_ex3_5 is starting this frame...
        if rect_ex3_5.status == NOT_STARTED and tThisFlip >= 12-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_5.frameNStart = frameN  # exact frame index
            rect_ex3_5.tStart = t  # local t and not account for scr refresh
            rect_ex3_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_5.started')
            # update status
            rect_ex3_5.status = STARTED
            rect_ex3_5.setAutoDraw(True)
        
        # if rect_ex3_5 is active this frame...
        if rect_ex3_5.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_5 is stopping this frame...
        if rect_ex3_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_5.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_5.tStop = t  # not accounting for scr refresh
                rect_ex3_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_5.stopped')
                # update status
                rect_ex3_5.status = FINISHED
                rect_ex3_5.setAutoDraw(False)
        
        # *text_ex3_6* updates
        
        # if text_ex3_6 is starting this frame...
        if text_ex3_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_6.frameNStart = frameN  # exact frame index
            text_ex3_6.tStart = t  # local t and not account for scr refresh
            text_ex3_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_6.started')
            # update status
            text_ex3_6.status = STARTED
            text_ex3_6.setAutoDraw(True)
        
        # if text_ex3_6 is active this frame...
        if text_ex3_6.status == STARTED:
            # update params
            pass
        
        # if text_ex3_6 is stopping this frame...
        if text_ex3_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_6.tStop = t  # not accounting for scr refresh
                text_ex3_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_6.stopped')
                # update status
                text_ex3_6.status = FINISHED
                text_ex3_6.setAutoDraw(False)
        
        # *rect_ex3_6* updates
        
        # if rect_ex3_6 is starting this frame...
        if rect_ex3_6.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            rect_ex3_6.frameNStart = frameN  # exact frame index
            rect_ex3_6.tStart = t  # local t and not account for scr refresh
            rect_ex3_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rect_ex3_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'rect_ex3_6.started')
            # update status
            rect_ex3_6.status = STARTED
            rect_ex3_6.setAutoDraw(True)
        
        # if rect_ex3_6 is active this frame...
        if rect_ex3_6.status == STARTED:
            # update params
            pass
        
        # if rect_ex3_6 is stopping this frame...
        if rect_ex3_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > rect_ex3_6.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                rect_ex3_6.tStop = t  # not accounting for scr refresh
                rect_ex3_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rect_ex3_6.stopped')
                # update status
                rect_ex3_6.status = FINISHED
                rect_ex3_6.setAutoDraw(False)
        
        # *text_ex3_array* updates
        
        # if text_ex3_array is starting this frame...
        if text_ex3_array.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_array.frameNStart = frameN  # exact frame index
            text_ex3_array.tStart = t  # local t and not account for scr refresh
            text_ex3_array.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_array, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_array.started')
            # update status
            text_ex3_array.status = STARTED
            text_ex3_array.setAutoDraw(True)
        
        # if text_ex3_array is active this frame...
        if text_ex3_array.status == STARTED:
            # update params
            pass
        
        # if text_ex3_array is stopping this frame...
        if text_ex3_array.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_array.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_array.tStop = t  # not accounting for scr refresh
                text_ex3_array.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_array.stopped')
                # update status
                text_ex3_array.status = FINISHED
                text_ex3_array.setAutoDraw(False)
        
        # *key_resp_ex3* updates
        waitOnFlip = False
        
        # if key_resp_ex3 is starting this frame...
        if key_resp_ex3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_ex3.frameNStart = frameN  # exact frame index
            key_resp_ex3.tStart = t  # local t and not account for scr refresh
            key_resp_ex3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_ex3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_ex3.started')
            # update status
            key_resp_ex3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_ex3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_ex3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        
        # if key_resp_ex3 is stopping this frame...
        if key_resp_ex3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > key_resp_ex3.tStartRefresh + 18-frameTolerance:
                # keep track of stop time/frame for later
                key_resp_ex3.tStop = t  # not accounting for scr refresh
                key_resp_ex3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_ex3.stopped')
                # update status
                key_resp_ex3.status = FINISHED
                key_resp_ex3.status = FINISHED
        if key_resp_ex3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_ex3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_ex3_allKeys.extend(theseKeys)
            if len(_key_resp_ex3_allKeys):
                key_resp_ex3.keys = _key_resp_ex3_allKeys[-1].name  # just the last key pressed
                key_resp_ex3.rt = _key_resp_ex3_allKeys[-1].rt
                key_resp_ex3.duration = _key_resp_ex3_allKeys[-1].duration
        
        # *text_ex3_cue2* updates
        
        # if text_ex3_cue2 is starting this frame...
        if text_ex3_cue2.status == NOT_STARTED and tThisFlip >= 9-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_cue2.frameNStart = frameN  # exact frame index
            text_ex3_cue2.tStart = t  # local t and not account for scr refresh
            text_ex3_cue2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_cue2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_cue2.started')
            # update status
            text_ex3_cue2.status = STARTED
            text_ex3_cue2.setAutoDraw(True)
        
        # if text_ex3_cue2 is active this frame...
        if text_ex3_cue2.status == STARTED:
            # update params
            pass
        
        # if text_ex3_cue2 is stopping this frame...
        if text_ex3_cue2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_cue2.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_cue2.tStop = t  # not accounting for scr refresh
                text_ex3_cue2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_cue2.stopped')
                # update status
                text_ex3_cue2.status = FINISHED
                text_ex3_cue2.setAutoDraw(False)
        
        # *text_ex3_cue1* updates
        
        # if text_ex3_cue1 is starting this frame...
        if text_ex3_cue1.status == NOT_STARTED and tThisFlip >= 15-frameTolerance:
            # keep track of start time/frame for later
            text_ex3_cue1.frameNStart = frameN  # exact frame index
            text_ex3_cue1.tStart = t  # local t and not account for scr refresh
            text_ex3_cue1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_ex3_cue1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_ex3_cue1.started')
            # update status
            text_ex3_cue1.status = STARTED
            text_ex3_cue1.setAutoDraw(True)
        
        # if text_ex3_cue1 is active this frame...
        if text_ex3_cue1.status == STARTED:
            # update params
            pass
        
        # if text_ex3_cue1 is stopping this frame...
        if text_ex3_cue1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_ex3_cue1.tStartRefresh + 3-frameTolerance:
                # keep track of stop time/frame for later
                text_ex3_cue1.tStop = t  # not accounting for scr refresh
                text_ex3_cue1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex3_cue1.stopped')
                # update status
                text_ex3_cue1.status = FINISHED
                text_ex3_cue1.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ex3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ex3" ---
    for thisComponent in ex3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('ex3.stopped', globalClock.getTime())
    # check responses
    if key_resp_ex3.keys in ['', [], None]:  # No response was made
        key_resp_ex3.keys = None
    thisExp.addData('key_resp_ex3.keys',key_resp_ex3.keys)
    if key_resp_ex3.keys != None:  # we had a response
        thisExp.addData('key_resp_ex3.rt', key_resp_ex3.rt)
        thisExp.addData('key_resp_ex3.duration', key_resp_ex3.duration)
    thisExp.nextEntry()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-18.000000)
    
    # --- Prepare to start Routine "intro_train1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_train1.started', globalClock.getTime())
    key_resp_intro_train1.keys = []
    key_resp_intro_train1.rt = []
    _key_resp_intro_train1_allKeys = []
    # keep track of which components have finished
    intro_train1Components = [text_intro_train1, key_resp_intro_train1]
    for thisComponent in intro_train1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_train1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_train1* updates
        
        # if text_intro_train1 is starting this frame...
        if text_intro_train1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_train1.frameNStart = frameN  # exact frame index
            text_intro_train1.tStart = t  # local t and not account for scr refresh
            text_intro_train1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_train1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_train1.started')
            # update status
            text_intro_train1.status = STARTED
            text_intro_train1.setAutoDraw(True)
        
        # if text_intro_train1 is active this frame...
        if text_intro_train1.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_train1* updates
        waitOnFlip = False
        
        # if key_resp_intro_train1 is starting this frame...
        if key_resp_intro_train1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_train1.frameNStart = frameN  # exact frame index
            key_resp_intro_train1.tStart = t  # local t and not account for scr refresh
            key_resp_intro_train1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_train1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_train1.started')
            # update status
            key_resp_intro_train1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_train1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_train1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_train1.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_train1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_train1_allKeys.extend(theseKeys)
            if len(_key_resp_intro_train1_allKeys):
                key_resp_intro_train1.keys = _key_resp_intro_train1_allKeys[-1].name  # just the last key pressed
                key_resp_intro_train1.rt = _key_resp_intro_train1_allKeys[-1].rt
                key_resp_intro_train1.duration = _key_resp_intro_train1_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_train1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_train1" ---
    for thisComponent in intro_train1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_train1.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_train1.keys in ['', [], None]:  # No response was made
        key_resp_intro_train1.keys = None
    thisExp.addData('key_resp_intro_train1.keys',key_resp_intro_train1.keys)
    if key_resp_intro_train1.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_train1.rt', key_resp_intro_train1.rt)
        thisExp.addData('key_resp_intro_train1.duration', key_resp_intro_train1.duration)
    thisExp.nextEntry()
    # the Routine "intro_train1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "train1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('train1.started', globalClock.getTime())
    key_resp_train1.keys = []
    key_resp_train1.rt = []
    _key_resp_train1_allKeys = []
    # Run 'Begin Routine' code from code_train1
    outlet.push_sample([f"begin_train1"])
    #for i in range(5): # Marks start of experiment
    #    # arduino.write(b"p")
    #    outlet.push_sample([f"sync_{i+1}"])
    #    time.sleep(0.5)
    
    next_letter_onset = core.getTime() + stimulus_duration + isi
    last_letter_onset = None
    offset_time = next_letter_onset + stimulus_duration
    trial_letters = []
    letter = np.random.choice(all_letters)
    N_train = 1
    last_is_target = False
    corrects_train1 = 0
    log.exp(f"Beggining of train1, N = {N_train}")
    feedback_text = ""
    last_len_pressed = 0
    
    # keep track of which components have finished
    train1Components = [text_train1, key_resp_train1, text_train1_feedback]
    for thisComponent in train1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "train1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_train1* updates
        
        # if text_train1 is starting this frame...
        if text_train1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_train1.frameNStart = frameN  # exact frame index
            text_train1.tStart = t  # local t and not account for scr refresh
            text_train1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_train1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_train1.started')
            # update status
            text_train1.status = STARTED
            text_train1.setAutoDraw(True)
        
        # if text_train1 is active this frame...
        if text_train1.status == STARTED:
            # update params
            text_train1.setText(letter, log=False)
        
        # *key_resp_train1* updates
        waitOnFlip = False
        
        # if key_resp_train1 is starting this frame...
        if key_resp_train1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_train1.frameNStart = frameN  # exact frame index
            key_resp_train1.tStart = t  # local t and not account for scr refresh
            key_resp_train1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_train1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_train1.started')
            # update status
            key_resp_train1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_train1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_train1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_train1.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_train1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_train1_allKeys.extend(theseKeys)
            if len(_key_resp_train1_allKeys):
                key_resp_train1.keys = [key.name for key in _key_resp_train1_allKeys]  # storing all keys
                key_resp_train1.rt = [key.rt for key in _key_resp_train1_allKeys]
                key_resp_train1.duration = [key.duration for key in _key_resp_train1_allKeys]
        
        # *text_train1_feedback* updates
        
        # if text_train1_feedback is starting this frame...
        if text_train1_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_train1_feedback.frameNStart = frameN  # exact frame index
            text_train1_feedback.tStart = t  # local t and not account for scr refresh
            text_train1_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_train1_feedback, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_train1_feedback.started')
            # update status
            text_train1_feedback.status = STARTED
            text_train1_feedback.setAutoDraw(True)
        
        # if text_train1_feedback is active this frame...
        if text_train1_feedback.status == STARTED:
            # update params
            text_train1_feedback.setText(feedback_text, log=False)
        # Run 'Each Frame' code from code_train1
        if key_resp_train1.keys:
            log.data(len(key_resp_train1.keys))
            if key_resp_train1.keys[-1] == "space" and len(key_resp_train1.keys) > last_len_pressed:
                if last_is_target:
                    feedback_text = "¡Correcto!"
        #            feedback_text = f"{key_resp_train1.keys}"
                    text_train1_feedback.color = "limegreen"
                    corrects_train1 += 1
                else:
                    feedback_text = "¡Incorrecto!"
        #            feedback_text = f"{key_resp_train1.keys}"
                    text_train1_feedback.color = "red"
                last_len_pressed += 1
            
        if core.getTime() >= next_letter_onset:
            last_letter_onset = next_letter_onset
            if len(trial_letters) > N_train:
                letter_nback = trial_letters[-N_train]
                if np.random.rand() < p_target:
                    letter = letter_nback # Letter is same as nback
                    log.exp(f"Nback letter: {letter_nback} TARGET")
                    last_is_target = True
                else:
                    # Letter is chosen at random from a list of all leters that are 
                    # not the nback letter.
                    letter = np.random.choice([i for i in all_letters if i is not letter_nback])
                    log.exp(f"Nback letter: {letter_nback} NON-TARGET")
                    last_is_target = False
            else:
                letter = np.random.choice(all_letters)
            trial_letters.append(letter)
            win.flip()
            log.exp(f"Stimulus: {letter}")
            
            next_letter_onset = core.getTime() + stimulus_duration + isi
            offset_time = last_letter_onset + stimulus_duration
        
        if core.getTime() >= offset_time:
            letter = ""
            feedback_text = ""
            win.flip()
            text_train1_feedback.color = "transparent"
        #    log.exp("Stimulus offset")
        
        if corrects_train1 >= 5:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in train1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "train1" ---
    for thisComponent in train1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('train1.stopped', globalClock.getTime())
    # check responses
    if key_resp_train1.keys in ['', [], None]:  # No response was made
        key_resp_train1.keys = None
    thisExp.addData('key_resp_train1.keys',key_resp_train1.keys)
    if key_resp_train1.keys != None:  # we had a response
        thisExp.addData('key_resp_train1.rt', key_resp_train1.rt)
        thisExp.addData('key_resp_train1.duration', key_resp_train1.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from code_train1
    outlet.push_sample([f"end_train1"])
    # for i in range(4): # Marks start of experiment
        # arduino.write(b"p")
    #    outlet.push_sample([f"sync_{i+1}"])
    #    time.sleep(0.5)
    # the Routine "train1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_train2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_train2.started', globalClock.getTime())
    key_resp_intro_train2.keys = []
    key_resp_intro_train2.rt = []
    _key_resp_intro_train2_allKeys = []
    # keep track of which components have finished
    intro_train2Components = [text_intro_train2, key_resp_intro_train2]
    for thisComponent in intro_train2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_train2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_train2* updates
        
        # if text_intro_train2 is starting this frame...
        if text_intro_train2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_train2.frameNStart = frameN  # exact frame index
            text_intro_train2.tStart = t  # local t and not account for scr refresh
            text_intro_train2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_train2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_train2.started')
            # update status
            text_intro_train2.status = STARTED
            text_intro_train2.setAutoDraw(True)
        
        # if text_intro_train2 is active this frame...
        if text_intro_train2.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_train2* updates
        waitOnFlip = False
        
        # if key_resp_intro_train2 is starting this frame...
        if key_resp_intro_train2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_train2.frameNStart = frameN  # exact frame index
            key_resp_intro_train2.tStart = t  # local t and not account for scr refresh
            key_resp_intro_train2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_train2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_train2.started')
            # update status
            key_resp_intro_train2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_train2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_train2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_train2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_train2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_train2_allKeys.extend(theseKeys)
            if len(_key_resp_intro_train2_allKeys):
                key_resp_intro_train2.keys = _key_resp_intro_train2_allKeys[-1].name  # just the last key pressed
                key_resp_intro_train2.rt = _key_resp_intro_train2_allKeys[-1].rt
                key_resp_intro_train2.duration = _key_resp_intro_train2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_train2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_train2" ---
    for thisComponent in intro_train2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_train2.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_train2.keys in ['', [], None]:  # No response was made
        key_resp_intro_train2.keys = None
    thisExp.addData('key_resp_intro_train2.keys',key_resp_intro_train2.keys)
    if key_resp_intro_train2.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_train2.rt', key_resp_intro_train2.rt)
        thisExp.addData('key_resp_intro_train2.duration', key_resp_intro_train2.duration)
    thisExp.nextEntry()
    # the Routine "intro_train2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "train2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('train2.started', globalClock.getTime())
    key_resp_train2.keys = []
    key_resp_train2.rt = []
    _key_resp_train2_allKeys = []
    # Run 'Begin Routine' code from code_train2
    outlet.push_sample([f"begin_train2"])
    #for i in range(5): # Marks start of experiment
    #    # arduino.write(b"p")
    #    outlet.push_sample([f"sync_{i+1}"])
    #    time.sleep(0.5)
    
    next_letter_onset = core.getTime() + stimulus_duration + isi
    last_letter_onset = None
    offset_time = next_letter_onset + stimulus_duration
    trial_letters = []
    letter = np.random.choice(all_letters)
    N_train = 2
    last_is_target = False
    corrects_train2 = 0
    log.exp(f"Beggining of train1, N = {N_train}")
    feedback_text = ""
    last_len_pressed = 0
    
    # keep track of which components have finished
    train2Components = [text_train2, key_resp_train2, text_train2_feedback]
    for thisComponent in train2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "train2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_train2* updates
        
        # if text_train2 is starting this frame...
        if text_train2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_train2.frameNStart = frameN  # exact frame index
            text_train2.tStart = t  # local t and not account for scr refresh
            text_train2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_train2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_train2.started')
            # update status
            text_train2.status = STARTED
            text_train2.setAutoDraw(True)
        
        # if text_train2 is active this frame...
        if text_train2.status == STARTED:
            # update params
            text_train2.setText(letter, log=False)
        
        # *key_resp_train2* updates
        waitOnFlip = False
        
        # if key_resp_train2 is starting this frame...
        if key_resp_train2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_train2.frameNStart = frameN  # exact frame index
            key_resp_train2.tStart = t  # local t and not account for scr refresh
            key_resp_train2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_train2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_train2.started')
            # update status
            key_resp_train2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_train2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_train2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_train2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_train2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_train2_allKeys.extend(theseKeys)
            if len(_key_resp_train2_allKeys):
                key_resp_train2.keys = [key.name for key in _key_resp_train2_allKeys]  # storing all keys
                key_resp_train2.rt = [key.rt for key in _key_resp_train2_allKeys]
                key_resp_train2.duration = [key.duration for key in _key_resp_train2_allKeys]
        
        # *text_train2_feedback* updates
        
        # if text_train2_feedback is starting this frame...
        if text_train2_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_train2_feedback.frameNStart = frameN  # exact frame index
            text_train2_feedback.tStart = t  # local t and not account for scr refresh
            text_train2_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_train2_feedback, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_train2_feedback.started')
            # update status
            text_train2_feedback.status = STARTED
            text_train2_feedback.setAutoDraw(True)
        
        # if text_train2_feedback is active this frame...
        if text_train2_feedback.status == STARTED:
            # update params
            text_train2_feedback.setText(feedback_text, log=False)
        # Run 'Each Frame' code from code_train2
        if key_resp_train2.keys:
            log.data(len(key_resp_train2.keys))
            if key_resp_train2.keys[-1] == "space" and len(key_resp_train2.keys) > last_len_pressed:
                if last_is_target:
                    feedback_text = "¡Correcto!"
        #            feedback_text = f"{key_resp_train1.keys}"
                    text_train2_feedback.color = "limegreen"
                    corrects_train2 += 1
                else:
                    feedback_text = "¡Incorrecto!"
        #            feedback_text = f"{key_resp_train1.keys}"
                    text_train2_feedback.color = "red"
                last_len_pressed += 1
            
        if core.getTime() >= next_letter_onset:
            last_letter_onset = next_letter_onset
            if len(trial_letters) > N_train:
                letter_nback = trial_letters[-N_train]
                if np.random.rand() < .3:
                    letter = letter_nback # Letter is same as nback
                    log.exp(f"Nback letter: {letter_nback} TARGET")
                    last_is_target = True
                else:
                    # Letter is chosen at random from a list of all leters that are 
                    # not the nback letter.
                    letter = np.random.choice([i for i in all_letters if i is not letter_nback])
                    log.exp(f"Nback letter: {letter_nback} NON-TARGET")
                    last_is_target = False
            else:
                letter = np.random.choice(all_letters)
            trial_letters.append(letter)
            win.flip()
            log.exp(f"Stimulus: {letter}")
            
            next_letter_onset = core.getTime() + stimulus_duration + isi
            offset_time = last_letter_onset + stimulus_duration
        
        if core.getTime() >= offset_time:
            letter = ""
            feedback_text = ""
            win.flip()
            text_train1_feedback.color = "transparent"
        #    log.exp("Stimulus offset")
        
        if corrects_train2 >= 5:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in train2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "train2" ---
    for thisComponent in train2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('train2.stopped', globalClock.getTime())
    # check responses
    if key_resp_train2.keys in ['', [], None]:  # No response was made
        key_resp_train2.keys = None
    thisExp.addData('key_resp_train2.keys',key_resp_train2.keys)
    if key_resp_train2.keys != None:  # we had a response
        thisExp.addData('key_resp_train2.rt', key_resp_train2.rt)
        thisExp.addData('key_resp_train2.duration', key_resp_train2.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from code_train2
    outlet.push_sample([f"end_train2"])
    # for i in range(4): # Marks start of experiment
        # arduino.write(b"p")
    #    outlet.push_sample([f"sync_{i+1}"])
    #    time.sleep(0.5)
    # the Routine "train2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "intro_nasatlx" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('intro_nasatlx.started', globalClock.getTime())
    key_resp_intro_nasatlx.keys = []
    key_resp_intro_nasatlx.rt = []
    _key_resp_intro_nasatlx_allKeys = []
    # Run 'Begin Routine' code from code_intro_nasatlx
    win.mouseVisible = True
    # keep track of which components have finished
    intro_nasatlxComponents = [text_intro_nasatlx, text_intro_nasatlx_1, text_intro_nasatlx_2, text_intro_nasatlx_3, key_resp_intro_nasatlx]
    for thisComponent in intro_nasatlxComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "intro_nasatlx" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_intro_nasatlx* updates
        
        # if text_intro_nasatlx is starting this frame...
        if text_intro_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_nasatlx.frameNStart = frameN  # exact frame index
            text_intro_nasatlx.tStart = t  # local t and not account for scr refresh
            text_intro_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_nasatlx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_nasatlx.started')
            # update status
            text_intro_nasatlx.status = STARTED
            text_intro_nasatlx.setAutoDraw(True)
        
        # if text_intro_nasatlx is active this frame...
        if text_intro_nasatlx.status == STARTED:
            # update params
            pass
        
        # *text_intro_nasatlx_1* updates
        
        # if text_intro_nasatlx_1 is starting this frame...
        if text_intro_nasatlx_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_nasatlx_1.frameNStart = frameN  # exact frame index
            text_intro_nasatlx_1.tStart = t  # local t and not account for scr refresh
            text_intro_nasatlx_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_nasatlx_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_nasatlx_1.started')
            # update status
            text_intro_nasatlx_1.status = STARTED
            text_intro_nasatlx_1.setAutoDraw(True)
        
        # if text_intro_nasatlx_1 is active this frame...
        if text_intro_nasatlx_1.status == STARTED:
            # update params
            pass
        
        # *text_intro_nasatlx_2* updates
        
        # if text_intro_nasatlx_2 is starting this frame...
        if text_intro_nasatlx_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_nasatlx_2.frameNStart = frameN  # exact frame index
            text_intro_nasatlx_2.tStart = t  # local t and not account for scr refresh
            text_intro_nasatlx_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_nasatlx_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_nasatlx_2.started')
            # update status
            text_intro_nasatlx_2.status = STARTED
            text_intro_nasatlx_2.setAutoDraw(True)
        
        # if text_intro_nasatlx_2 is active this frame...
        if text_intro_nasatlx_2.status == STARTED:
            # update params
            pass
        
        # *text_intro_nasatlx_3* updates
        
        # if text_intro_nasatlx_3 is starting this frame...
        if text_intro_nasatlx_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_intro_nasatlx_3.frameNStart = frameN  # exact frame index
            text_intro_nasatlx_3.tStart = t  # local t and not account for scr refresh
            text_intro_nasatlx_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_intro_nasatlx_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_intro_nasatlx_3.started')
            # update status
            text_intro_nasatlx_3.status = STARTED
            text_intro_nasatlx_3.setAutoDraw(True)
        
        # if text_intro_nasatlx_3 is active this frame...
        if text_intro_nasatlx_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_intro_nasatlx* updates
        waitOnFlip = False
        
        # if key_resp_intro_nasatlx is starting this frame...
        if key_resp_intro_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_intro_nasatlx.frameNStart = frameN  # exact frame index
            key_resp_intro_nasatlx.tStart = t  # local t and not account for scr refresh
            key_resp_intro_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_intro_nasatlx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_intro_nasatlx.started')
            # update status
            key_resp_intro_nasatlx.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_intro_nasatlx.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_intro_nasatlx.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_intro_nasatlx.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_intro_nasatlx.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_intro_nasatlx_allKeys.extend(theseKeys)
            if len(_key_resp_intro_nasatlx_allKeys):
                key_resp_intro_nasatlx.keys = _key_resp_intro_nasatlx_allKeys[-1].name  # just the last key pressed
                key_resp_intro_nasatlx.rt = _key_resp_intro_nasatlx_allKeys[-1].rt
                key_resp_intro_nasatlx.duration = _key_resp_intro_nasatlx_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in intro_nasatlxComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "intro_nasatlx" ---
    for thisComponent in intro_nasatlxComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('intro_nasatlx.stopped', globalClock.getTime())
    # check responses
    if key_resp_intro_nasatlx.keys in ['', [], None]:  # No response was made
        key_resp_intro_nasatlx.keys = None
    thisExp.addData('key_resp_intro_nasatlx.keys',key_resp_intro_nasatlx.keys)
    if key_resp_intro_nasatlx.keys != None:  # we had a response
        thisExp.addData('key_resp_intro_nasatlx.rt', key_resp_intro_nasatlx.rt)
        thisExp.addData('key_resp_intro_nasatlx.duration', key_resp_intro_nasatlx.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from code_intro_nasatlx
    win.mouseVisible = False
    # the Routine "intro_nasatlx" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "nasaTLX" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('nasaTLX.started', globalClock.getTime())
    slider_mental_demand.reset()
    slider_physical_demand.reset()
    slider_temporal_demand.reset()
    slider_effort.reset()
    slider_performance.reset()
    slider_frustration.reset()
    key_resp_nasatlx.keys = []
    key_resp_nasatlx.rt = []
    _key_resp_nasatlx_allKeys = []
    # Run 'Begin Routine' code from code_nasatlx
    win.mouseVisible = True
    # keep track of which components have finished
    nasaTLXComponents = [text_mental_demand, slider_mental_demand, text_physical_demand, slider_physical_demand, text_temporal_demand, slider_temporal_demand, text_effort, slider_effort, text_performance, slider_performance, text_frustration, slider_frustration, text_nasatlx, key_resp_nasatlx]
    for thisComponent in nasaTLXComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "nasaTLX" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_mental_demand* updates
        
        # if text_mental_demand is starting this frame...
        if text_mental_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_mental_demand.frameNStart = frameN  # exact frame index
            text_mental_demand.tStart = t  # local t and not account for scr refresh
            text_mental_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_mental_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_mental_demand.started')
            # update status
            text_mental_demand.status = STARTED
            text_mental_demand.setAutoDraw(True)
        
        # if text_mental_demand is active this frame...
        if text_mental_demand.status == STARTED:
            # update params
            pass
        
        # *slider_mental_demand* updates
        
        # if slider_mental_demand is starting this frame...
        if slider_mental_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_mental_demand.frameNStart = frameN  # exact frame index
            slider_mental_demand.tStart = t  # local t and not account for scr refresh
            slider_mental_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_mental_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_mental_demand.started')
            # update status
            slider_mental_demand.status = STARTED
            slider_mental_demand.setAutoDraw(True)
        
        # if slider_mental_demand is active this frame...
        if slider_mental_demand.status == STARTED:
            # update params
            pass
        
        # *text_physical_demand* updates
        
        # if text_physical_demand is starting this frame...
        if text_physical_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_physical_demand.frameNStart = frameN  # exact frame index
            text_physical_demand.tStart = t  # local t and not account for scr refresh
            text_physical_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_physical_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_physical_demand.started')
            # update status
            text_physical_demand.status = STARTED
            text_physical_demand.setAutoDraw(True)
        
        # if text_physical_demand is active this frame...
        if text_physical_demand.status == STARTED:
            # update params
            pass
        
        # *slider_physical_demand* updates
        
        # if slider_physical_demand is starting this frame...
        if slider_physical_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_physical_demand.frameNStart = frameN  # exact frame index
            slider_physical_demand.tStart = t  # local t and not account for scr refresh
            slider_physical_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_physical_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_physical_demand.started')
            # update status
            slider_physical_demand.status = STARTED
            slider_physical_demand.setAutoDraw(True)
        
        # if slider_physical_demand is active this frame...
        if slider_physical_demand.status == STARTED:
            # update params
            pass
        
        # *text_temporal_demand* updates
        
        # if text_temporal_demand is starting this frame...
        if text_temporal_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_temporal_demand.frameNStart = frameN  # exact frame index
            text_temporal_demand.tStart = t  # local t and not account for scr refresh
            text_temporal_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_temporal_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_temporal_demand.started')
            # update status
            text_temporal_demand.status = STARTED
            text_temporal_demand.setAutoDraw(True)
        
        # if text_temporal_demand is active this frame...
        if text_temporal_demand.status == STARTED:
            # update params
            pass
        
        # *slider_temporal_demand* updates
        
        # if slider_temporal_demand is starting this frame...
        if slider_temporal_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_temporal_demand.frameNStart = frameN  # exact frame index
            slider_temporal_demand.tStart = t  # local t and not account for scr refresh
            slider_temporal_demand.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_temporal_demand, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_temporal_demand.started')
            # update status
            slider_temporal_demand.status = STARTED
            slider_temporal_demand.setAutoDraw(True)
        
        # if slider_temporal_demand is active this frame...
        if slider_temporal_demand.status == STARTED:
            # update params
            pass
        
        # *text_effort* updates
        
        # if text_effort is starting this frame...
        if text_effort.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_effort.frameNStart = frameN  # exact frame index
            text_effort.tStart = t  # local t and not account for scr refresh
            text_effort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_effort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_effort.started')
            # update status
            text_effort.status = STARTED
            text_effort.setAutoDraw(True)
        
        # if text_effort is active this frame...
        if text_effort.status == STARTED:
            # update params
            pass
        
        # *slider_effort* updates
        
        # if slider_effort is starting this frame...
        if slider_effort.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_effort.frameNStart = frameN  # exact frame index
            slider_effort.tStart = t  # local t and not account for scr refresh
            slider_effort.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_effort, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_effort.started')
            # update status
            slider_effort.status = STARTED
            slider_effort.setAutoDraw(True)
        
        # if slider_effort is active this frame...
        if slider_effort.status == STARTED:
            # update params
            pass
        
        # *text_performance* updates
        
        # if text_performance is starting this frame...
        if text_performance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_performance.frameNStart = frameN  # exact frame index
            text_performance.tStart = t  # local t and not account for scr refresh
            text_performance.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_performance, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_performance.started')
            # update status
            text_performance.status = STARTED
            text_performance.setAutoDraw(True)
        
        # if text_performance is active this frame...
        if text_performance.status == STARTED:
            # update params
            pass
        
        # *slider_performance* updates
        
        # if slider_performance is starting this frame...
        if slider_performance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_performance.frameNStart = frameN  # exact frame index
            slider_performance.tStart = t  # local t and not account for scr refresh
            slider_performance.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_performance, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_performance.started')
            # update status
            slider_performance.status = STARTED
            slider_performance.setAutoDraw(True)
        
        # if slider_performance is active this frame...
        if slider_performance.status == STARTED:
            # update params
            pass
        
        # *text_frustration* updates
        
        # if text_frustration is starting this frame...
        if text_frustration.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_frustration.frameNStart = frameN  # exact frame index
            text_frustration.tStart = t  # local t and not account for scr refresh
            text_frustration.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_frustration, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_frustration.started')
            # update status
            text_frustration.status = STARTED
            text_frustration.setAutoDraw(True)
        
        # if text_frustration is active this frame...
        if text_frustration.status == STARTED:
            # update params
            pass
        
        # *slider_frustration* updates
        
        # if slider_frustration is starting this frame...
        if slider_frustration.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_frustration.frameNStart = frameN  # exact frame index
            slider_frustration.tStart = t  # local t and not account for scr refresh
            slider_frustration.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_frustration, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_frustration.started')
            # update status
            slider_frustration.status = STARTED
            slider_frustration.setAutoDraw(True)
        
        # if slider_frustration is active this frame...
        if slider_frustration.status == STARTED:
            # update params
            pass
        
        # *text_nasatlx* updates
        
        # if text_nasatlx is starting this frame...
        if text_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_nasatlx.frameNStart = frameN  # exact frame index
            text_nasatlx.tStart = t  # local t and not account for scr refresh
            text_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_nasatlx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_nasatlx.started')
            # update status
            text_nasatlx.status = STARTED
            text_nasatlx.setAutoDraw(True)
        
        # if text_nasatlx is active this frame...
        if text_nasatlx.status == STARTED:
            # update params
            pass
        
        # *key_resp_nasatlx* updates
        waitOnFlip = False
        
        # if key_resp_nasatlx is starting this frame...
        if key_resp_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_nasatlx.frameNStart = frameN  # exact frame index
            key_resp_nasatlx.tStart = t  # local t and not account for scr refresh
            key_resp_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_nasatlx, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_nasatlx.started')
            # update status
            key_resp_nasatlx.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_nasatlx.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_nasatlx.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_nasatlx.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_nasatlx.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_nasatlx_allKeys.extend(theseKeys)
            if len(_key_resp_nasatlx_allKeys):
                key_resp_nasatlx.keys = _key_resp_nasatlx_allKeys[-1].name  # just the last key pressed
                key_resp_nasatlx.rt = _key_resp_nasatlx_allKeys[-1].rt
                key_resp_nasatlx.duration = _key_resp_nasatlx_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in nasaTLXComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "nasaTLX" ---
    for thisComponent in nasaTLXComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('nasaTLX.stopped', globalClock.getTime())
    thisExp.addData('slider_mental_demand.response', slider_mental_demand.getRating())
    thisExp.addData('slider_mental_demand.rt', slider_mental_demand.getRT())
    thisExp.addData('slider_physical_demand.response', slider_physical_demand.getRating())
    thisExp.addData('slider_physical_demand.rt', slider_physical_demand.getRT())
    thisExp.addData('slider_temporal_demand.response', slider_temporal_demand.getRating())
    thisExp.addData('slider_temporal_demand.rt', slider_temporal_demand.getRT())
    thisExp.addData('slider_effort.response', slider_effort.getRating())
    thisExp.addData('slider_effort.rt', slider_effort.getRT())
    thisExp.addData('slider_performance.response', slider_performance.getRating())
    thisExp.addData('slider_performance.rt', slider_performance.getRT())
    thisExp.addData('slider_frustration.response', slider_frustration.getRating())
    thisExp.addData('slider_frustration.rt', slider_frustration.getRT())
    # check responses
    if key_resp_nasatlx.keys in ['', [], None]:  # No response was made
        key_resp_nasatlx.keys = None
    thisExp.addData('key_resp_nasatlx.keys',key_resp_nasatlx.keys)
    if key_resp_nasatlx.keys != None:  # we had a response
        thisExp.addData('key_resp_nasatlx.rt', key_resp_nasatlx.rt)
        thisExp.addData('key_resp_nasatlx.duration', key_resp_nasatlx.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from code_nasatlx
    win.mouseVisible = False
    # the Routine "nasaTLX" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pretrial" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('pretrial.started', globalClock.getTime())
    key_resp_pretrial.keys = []
    key_resp_pretrial.rt = []
    _key_resp_pretrial_allKeys = []
    # keep track of which components have finished
    pretrialComponents = [text_pretrial, key_resp_pretrial]
    for thisComponent in pretrialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pretrial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_pretrial* updates
        
        # if text_pretrial is starting this frame...
        if text_pretrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pretrial.frameNStart = frameN  # exact frame index
            text_pretrial.tStart = t  # local t and not account for scr refresh
            text_pretrial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pretrial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pretrial.started')
            # update status
            text_pretrial.status = STARTED
            text_pretrial.setAutoDraw(True)
        
        # if text_pretrial is active this frame...
        if text_pretrial.status == STARTED:
            # update params
            pass
        
        # *key_resp_pretrial* updates
        waitOnFlip = False
        
        # if key_resp_pretrial is starting this frame...
        if key_resp_pretrial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_pretrial.frameNStart = frameN  # exact frame index
            key_resp_pretrial.tStart = t  # local t and not account for scr refresh
            key_resp_pretrial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_pretrial, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_pretrial.started')
            # update status
            key_resp_pretrial.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_pretrial.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_pretrial.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_pretrial.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_pretrial.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_pretrial_allKeys.extend(theseKeys)
            if len(_key_resp_pretrial_allKeys):
                key_resp_pretrial.keys = _key_resp_pretrial_allKeys[-1].name  # just the last key pressed
                key_resp_pretrial.rt = _key_resp_pretrial_allKeys[-1].rt
                key_resp_pretrial.duration = _key_resp_pretrial_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pretrialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pretrial" ---
    for thisComponent in pretrialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('pretrial.stopped', globalClock.getTime())
    # check responses
    if key_resp_pretrial.keys in ['', [], None]:  # No response was made
        key_resp_pretrial.keys = None
    thisExp.addData('key_resp_pretrial.keys',key_resp_pretrial.keys)
    if key_resp_pretrial.keys != None:  # we had a response
        thisExp.addData('key_resp_pretrial.rt', key_resp_pretrial.rt)
        thisExp.addData('key_resp_pretrial.duration', key_resp_pretrial.duration)
    thisExp.nextEntry()
    # the Routine "pretrial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler(nReps=3.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('blocks.xlsx'),
        seed=None, name='blocks')
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "displayn" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('displayn.started', globalClock.getTime())
        text_displayn.setText(f"""Éste es el comienzo del bloque {blocks.thisN+1}
        
        Presiona la barra espaciadora si la letra que aparece es igual a la letra que aparece {N} letras antes.
        
        N = {N}
        
        [Presiona la barra espaciadora para continuar...]
        """)
        key_resp_displayn.keys = []
        key_resp_displayn.rt = []
        _key_resp_displayn_allKeys = []
        # keep track of which components have finished
        displaynComponents = [text_displayn, key_resp_displayn]
        for thisComponent in displaynComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "displayn" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_displayn* updates
            
            # if text_displayn is starting this frame...
            if text_displayn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_displayn.frameNStart = frameN  # exact frame index
                text_displayn.tStart = t  # local t and not account for scr refresh
                text_displayn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_displayn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_displayn.started')
                # update status
                text_displayn.status = STARTED
                text_displayn.setAutoDraw(True)
            
            # if text_displayn is active this frame...
            if text_displayn.status == STARTED:
                # update params
                pass
            
            # *key_resp_displayn* updates
            waitOnFlip = False
            
            # if key_resp_displayn is starting this frame...
            if key_resp_displayn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_displayn.frameNStart = frameN  # exact frame index
                key_resp_displayn.tStart = t  # local t and not account for scr refresh
                key_resp_displayn.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_displayn, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_displayn.started')
                # update status
                key_resp_displayn.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_displayn.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_displayn.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_displayn.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_displayn.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_displayn_allKeys.extend(theseKeys)
                if len(_key_resp_displayn_allKeys):
                    key_resp_displayn.keys = _key_resp_displayn_allKeys[-1].name  # just the last key pressed
                    key_resp_displayn.rt = _key_resp_displayn_allKeys[-1].rt
                    key_resp_displayn.duration = _key_resp_displayn_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in displaynComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "displayn" ---
        for thisComponent in displaynComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('displayn.stopped', globalClock.getTime())
        # check responses
        if key_resp_displayn.keys in ['', [], None]:  # No response was made
            key_resp_displayn.keys = None
        blocks.addData('key_resp_displayn.keys',key_resp_displayn.keys)
        if key_resp_displayn.keys != None:  # we had a response
            blocks.addData('key_resp_displayn.rt', key_resp_displayn.rt)
            blocks.addData('key_resp_displayn.duration', key_resp_displayn.duration)
        # the Routine "displayn" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fix" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fix.started', globalClock.getTime())
        # keep track of which components have finished
        fixComponents = [cross, text_nothing]
        for thisComponent in fixComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fix" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross* updates
            
            # if cross is starting this frame...
            if cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross.frameNStart = frameN  # exact frame index
                cross.tStart = t  # local t and not account for scr refresh
                cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross.started')
                # update status
                cross.status = STARTED
                cross.setAutoDraw(True)
            
            # if cross is active this frame...
            if cross.status == STARTED:
                # update params
                pass
            
            # if cross is stopping this frame...
            if cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    cross.tStop = t  # not accounting for scr refresh
                    cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross.stopped')
                    # update status
                    cross.status = FINISHED
                    cross.setAutoDraw(False)
            
            # *text_nothing* updates
            
            # if text_nothing is starting this frame...
            if text_nothing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_nothing.frameNStart = frameN  # exact frame index
                text_nothing.tStart = t  # local t and not account for scr refresh
                text_nothing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_nothing, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_nothing.started')
                # update status
                text_nothing.status = STARTED
                text_nothing.setAutoDraw(True)
            
            # if text_nothing is active this frame...
            if text_nothing.status == STARTED:
                # update params
                pass
            
            # if text_nothing is stopping this frame...
            if text_nothing.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_nothing.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text_nothing.tStop = t  # not accounting for scr refresh
                    text_nothing.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_nothing.stopped')
                    # update status
                    text_nothing.status = FINISHED
                    text_nothing.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix" ---
        for thisComponent in fixComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fix.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trial.started', globalClock.getTime())
        key_resp_trial.keys = []
        key_resp_trial.rt = []
        _key_resp_trial_allKeys = []
        # Run 'Begin Routine' code from code_trial
        outlet.push_sample([f"begin_routine_{blocks.thisN+1}"])
        for i in range(5): # Marks start of experiment
            # arduino.write(b"p")
            outlet.push_sample([f"sync_{i+1}"])
            time.sleep(0.5)
        
        next_letter_onset = core.getTime() + stimulus_duration + isi
        last_letter_onset = None
        offset_time = next_letter_onset + stimulus_duration
        trial_letters = []
        letter = np.random.choice(all_letters)
        
        log.exp(f"Beggining of trial, N = {N}")
        # keep track of which components have finished
        trialComponents = [text_letter, key_resp_trial]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > 90-frameTolerance:
                continueRoutine = False
            
            # *text_letter* updates
            
            # if text_letter is starting this frame...
            if text_letter.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_letter.frameNStart = frameN  # exact frame index
                text_letter.tStart = t  # local t and not account for scr refresh
                text_letter.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_letter, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_letter.started')
                # update status
                text_letter.status = STARTED
                text_letter.setAutoDraw(True)
            
            # if text_letter is active this frame...
            if text_letter.status == STARTED:
                # update params
                text_letter.setText(letter, log=False)
            
            # if text_letter is stopping this frame...
            if text_letter.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_letter.tStartRefresh + 90-frameTolerance:
                    # keep track of stop time/frame for later
                    text_letter.tStop = t  # not accounting for scr refresh
                    text_letter.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_letter.stopped')
                    # update status
                    text_letter.status = FINISHED
                    text_letter.setAutoDraw(False)
            
            # *key_resp_trial* updates
            waitOnFlip = False
            
            # if key_resp_trial is starting this frame...
            if key_resp_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_trial.frameNStart = frameN  # exact frame index
                key_resp_trial.tStart = t  # local t and not account for scr refresh
                key_resp_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_trial.started')
                # update status
                key_resp_trial.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_trial.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_trial.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_trial.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_trial.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_trial_allKeys.extend(theseKeys)
                if len(_key_resp_trial_allKeys):
                    key_resp_trial.keys = _key_resp_trial_allKeys[-1].name  # just the last key pressed
                    key_resp_trial.rt = _key_resp_trial_allKeys[-1].rt
                    key_resp_trial.duration = _key_resp_trial_allKeys[-1].duration
            # Run 'Each Frame' code from code_trial
            if core.getTime() >= next_letter_onset:
                last_letter_onset = next_letter_onset
                if len(trial_letters) > N:
                    letter_nback = trial_letters[-N]
                    if np.random.rand() < p_target:
                        letter = letter_nback # Letter is same as nback
                        log.exp(f"Nback letter: {letter_nback} TARGET")
                    else:
                        # Letter is chosen at random from a list of all leters that are 
                        # not the nback letter.
                        letter = np.random.choice([i for i in all_letters if i is not letter_nback])
                        log.exp(f"Nback letter: {letter_nback} NON-TARGET")
                else:
                    letter = np.random.choice(all_letters)
                trial_letters.append(letter)
                win.flip()
                log.exp(f"Stimulus: {letter}")
                
                next_letter_onset = core.getTime() + stimulus_duration + isi
                offset_time = last_letter_onset + stimulus_duration
            
            if core.getTime() >= offset_time:
                letter = ""
                win.flip()
            #    log.exp("Stimulus offset")
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trial.stopped', globalClock.getTime())
        # check responses
        if key_resp_trial.keys in ['', [], None]:  # No response was made
            key_resp_trial.keys = None
        blocks.addData('key_resp_trial.keys',key_resp_trial.keys)
        if key_resp_trial.keys != None:  # we had a response
            blocks.addData('key_resp_trial.rt', key_resp_trial.rt)
            blocks.addData('key_resp_trial.duration', key_resp_trial.duration)
        # Run 'End Routine' code from code_trial
        outlet.push_sample([f"end_routine_{blocks.thisN+1}"])
        for i in range(4): # Marks start of experiment
            # arduino.write(b"p")
            outlet.push_sample([f"sync_{i+1}"])
            time.sleep(0.5)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "nasaTLX" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('nasaTLX.started', globalClock.getTime())
        slider_mental_demand.reset()
        slider_physical_demand.reset()
        slider_temporal_demand.reset()
        slider_effort.reset()
        slider_performance.reset()
        slider_frustration.reset()
        key_resp_nasatlx.keys = []
        key_resp_nasatlx.rt = []
        _key_resp_nasatlx_allKeys = []
        # Run 'Begin Routine' code from code_nasatlx
        win.mouseVisible = True
        # keep track of which components have finished
        nasaTLXComponents = [text_mental_demand, slider_mental_demand, text_physical_demand, slider_physical_demand, text_temporal_demand, slider_temporal_demand, text_effort, slider_effort, text_performance, slider_performance, text_frustration, slider_frustration, text_nasatlx, key_resp_nasatlx]
        for thisComponent in nasaTLXComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "nasaTLX" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_mental_demand* updates
            
            # if text_mental_demand is starting this frame...
            if text_mental_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_mental_demand.frameNStart = frameN  # exact frame index
                text_mental_demand.tStart = t  # local t and not account for scr refresh
                text_mental_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_mental_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_mental_demand.started')
                # update status
                text_mental_demand.status = STARTED
                text_mental_demand.setAutoDraw(True)
            
            # if text_mental_demand is active this frame...
            if text_mental_demand.status == STARTED:
                # update params
                pass
            
            # *slider_mental_demand* updates
            
            # if slider_mental_demand is starting this frame...
            if slider_mental_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_mental_demand.frameNStart = frameN  # exact frame index
                slider_mental_demand.tStart = t  # local t and not account for scr refresh
                slider_mental_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_mental_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_mental_demand.started')
                # update status
                slider_mental_demand.status = STARTED
                slider_mental_demand.setAutoDraw(True)
            
            # if slider_mental_demand is active this frame...
            if slider_mental_demand.status == STARTED:
                # update params
                pass
            
            # *text_physical_demand* updates
            
            # if text_physical_demand is starting this frame...
            if text_physical_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_physical_demand.frameNStart = frameN  # exact frame index
                text_physical_demand.tStart = t  # local t and not account for scr refresh
                text_physical_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_physical_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_physical_demand.started')
                # update status
                text_physical_demand.status = STARTED
                text_physical_demand.setAutoDraw(True)
            
            # if text_physical_demand is active this frame...
            if text_physical_demand.status == STARTED:
                # update params
                pass
            
            # *slider_physical_demand* updates
            
            # if slider_physical_demand is starting this frame...
            if slider_physical_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_physical_demand.frameNStart = frameN  # exact frame index
                slider_physical_demand.tStart = t  # local t and not account for scr refresh
                slider_physical_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_physical_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_physical_demand.started')
                # update status
                slider_physical_demand.status = STARTED
                slider_physical_demand.setAutoDraw(True)
            
            # if slider_physical_demand is active this frame...
            if slider_physical_demand.status == STARTED:
                # update params
                pass
            
            # *text_temporal_demand* updates
            
            # if text_temporal_demand is starting this frame...
            if text_temporal_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_temporal_demand.frameNStart = frameN  # exact frame index
                text_temporal_demand.tStart = t  # local t and not account for scr refresh
                text_temporal_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_temporal_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_temporal_demand.started')
                # update status
                text_temporal_demand.status = STARTED
                text_temporal_demand.setAutoDraw(True)
            
            # if text_temporal_demand is active this frame...
            if text_temporal_demand.status == STARTED:
                # update params
                pass
            
            # *slider_temporal_demand* updates
            
            # if slider_temporal_demand is starting this frame...
            if slider_temporal_demand.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_temporal_demand.frameNStart = frameN  # exact frame index
                slider_temporal_demand.tStart = t  # local t and not account for scr refresh
                slider_temporal_demand.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_temporal_demand, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_temporal_demand.started')
                # update status
                slider_temporal_demand.status = STARTED
                slider_temporal_demand.setAutoDraw(True)
            
            # if slider_temporal_demand is active this frame...
            if slider_temporal_demand.status == STARTED:
                # update params
                pass
            
            # *text_effort* updates
            
            # if text_effort is starting this frame...
            if text_effort.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_effort.frameNStart = frameN  # exact frame index
                text_effort.tStart = t  # local t and not account for scr refresh
                text_effort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_effort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_effort.started')
                # update status
                text_effort.status = STARTED
                text_effort.setAutoDraw(True)
            
            # if text_effort is active this frame...
            if text_effort.status == STARTED:
                # update params
                pass
            
            # *slider_effort* updates
            
            # if slider_effort is starting this frame...
            if slider_effort.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_effort.frameNStart = frameN  # exact frame index
                slider_effort.tStart = t  # local t and not account for scr refresh
                slider_effort.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_effort, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_effort.started')
                # update status
                slider_effort.status = STARTED
                slider_effort.setAutoDraw(True)
            
            # if slider_effort is active this frame...
            if slider_effort.status == STARTED:
                # update params
                pass
            
            # *text_performance* updates
            
            # if text_performance is starting this frame...
            if text_performance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_performance.frameNStart = frameN  # exact frame index
                text_performance.tStart = t  # local t and not account for scr refresh
                text_performance.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_performance, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_performance.started')
                # update status
                text_performance.status = STARTED
                text_performance.setAutoDraw(True)
            
            # if text_performance is active this frame...
            if text_performance.status == STARTED:
                # update params
                pass
            
            # *slider_performance* updates
            
            # if slider_performance is starting this frame...
            if slider_performance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_performance.frameNStart = frameN  # exact frame index
                slider_performance.tStart = t  # local t and not account for scr refresh
                slider_performance.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_performance, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_performance.started')
                # update status
                slider_performance.status = STARTED
                slider_performance.setAutoDraw(True)
            
            # if slider_performance is active this frame...
            if slider_performance.status == STARTED:
                # update params
                pass
            
            # *text_frustration* updates
            
            # if text_frustration is starting this frame...
            if text_frustration.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_frustration.frameNStart = frameN  # exact frame index
                text_frustration.tStart = t  # local t and not account for scr refresh
                text_frustration.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_frustration, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_frustration.started')
                # update status
                text_frustration.status = STARTED
                text_frustration.setAutoDraw(True)
            
            # if text_frustration is active this frame...
            if text_frustration.status == STARTED:
                # update params
                pass
            
            # *slider_frustration* updates
            
            # if slider_frustration is starting this frame...
            if slider_frustration.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_frustration.frameNStart = frameN  # exact frame index
                slider_frustration.tStart = t  # local t and not account for scr refresh
                slider_frustration.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_frustration, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_frustration.started')
                # update status
                slider_frustration.status = STARTED
                slider_frustration.setAutoDraw(True)
            
            # if slider_frustration is active this frame...
            if slider_frustration.status == STARTED:
                # update params
                pass
            
            # *text_nasatlx* updates
            
            # if text_nasatlx is starting this frame...
            if text_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_nasatlx.frameNStart = frameN  # exact frame index
                text_nasatlx.tStart = t  # local t and not account for scr refresh
                text_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_nasatlx, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_nasatlx.started')
                # update status
                text_nasatlx.status = STARTED
                text_nasatlx.setAutoDraw(True)
            
            # if text_nasatlx is active this frame...
            if text_nasatlx.status == STARTED:
                # update params
                pass
            
            # *key_resp_nasatlx* updates
            waitOnFlip = False
            
            # if key_resp_nasatlx is starting this frame...
            if key_resp_nasatlx.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_nasatlx.frameNStart = frameN  # exact frame index
                key_resp_nasatlx.tStart = t  # local t and not account for scr refresh
                key_resp_nasatlx.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_nasatlx, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_nasatlx.started')
                # update status
                key_resp_nasatlx.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_nasatlx.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_nasatlx.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_nasatlx.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_nasatlx.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_nasatlx_allKeys.extend(theseKeys)
                if len(_key_resp_nasatlx_allKeys):
                    key_resp_nasatlx.keys = _key_resp_nasatlx_allKeys[-1].name  # just the last key pressed
                    key_resp_nasatlx.rt = _key_resp_nasatlx_allKeys[-1].rt
                    key_resp_nasatlx.duration = _key_resp_nasatlx_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in nasaTLXComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "nasaTLX" ---
        for thisComponent in nasaTLXComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('nasaTLX.stopped', globalClock.getTime())
        blocks.addData('slider_mental_demand.response', slider_mental_demand.getRating())
        blocks.addData('slider_mental_demand.rt', slider_mental_demand.getRT())
        blocks.addData('slider_physical_demand.response', slider_physical_demand.getRating())
        blocks.addData('slider_physical_demand.rt', slider_physical_demand.getRT())
        blocks.addData('slider_temporal_demand.response', slider_temporal_demand.getRating())
        blocks.addData('slider_temporal_demand.rt', slider_temporal_demand.getRT())
        blocks.addData('slider_effort.response', slider_effort.getRating())
        blocks.addData('slider_effort.rt', slider_effort.getRT())
        blocks.addData('slider_performance.response', slider_performance.getRating())
        blocks.addData('slider_performance.rt', slider_performance.getRT())
        blocks.addData('slider_frustration.response', slider_frustration.getRating())
        blocks.addData('slider_frustration.rt', slider_frustration.getRT())
        # check responses
        if key_resp_nasatlx.keys in ['', [], None]:  # No response was made
            key_resp_nasatlx.keys = None
        blocks.addData('key_resp_nasatlx.keys',key_resp_nasatlx.keys)
        if key_resp_nasatlx.keys != None:  # we had a response
            blocks.addData('key_resp_nasatlx.rt', key_resp_nasatlx.rt)
            blocks.addData('key_resp_nasatlx.duration', key_resp_nasatlx.duration)
        # Run 'End Routine' code from code_nasatlx
        win.mouseVisible = False
        # the Routine "nasaTLX" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 3.0 repeats of 'blocks'
    
    
    # --- Prepare to start Routine "end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('end.started', globalClock.getTime())
    # keep track of which components have finished
    endComponents = [text_end]
    for thisComponent in endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 4.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('end.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.000000)
    # Run 'End Experiment' code from code_train1
    # arduino.close()
    # Run 'End Experiment' code from code_train2
    # arduino.close()
    # Run 'End Experiment' code from code_trial
    # arduino.close()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
