import os
import sys
import numpy as np
import soundfile as sf
import torch
import yaml
from utils import get_model, load_weights

import sounddevice as sd

win = None

class AudioDirector:
    def __init__(self, mode="async"):
        self.mode = mode
        self.started=False
        self.async_index = 0
        self.last_as_idx = 0
        self.async_rep_index = 0
        self.vox_gain = 1.0
        self.instr_gain = 1.0

    def setup(self, cfgpath):
        self.config = None
        if cfgpath == None:
            print("No config file path provided, returning...")
            return
        if not os.path.exists(cfgpath):
            print("Config file does not exist, returning...")
            return
        with open(cfgpath, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
                self.chunkLength = self.config['separator']['chunk']
                self.hop = self.config['separator']['hop']
                winType = self.config['separator']['window']
                #match winType with np.winType
                if winType == "hanning" or winType =='hann':
                    win = np.hanning(self.chunkLength)
                elif winType == "hamming":
                    win = np.hamming(self.chunkLength)
                elif winType == "blackman":
                    win = np.blackman(self.chunkLength)
                elif winType == "bartlett":
                    win = np.bartlett(self.chunkLength)
                else:
                    print("Invalid or no window type - using no window")
                    win = None
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
        self.loadModel(self.config['global']['model'])
            
    def setTarget(self, source=None, target_sr=None):
        if self.mode == "async":
            if source == None:
                print("No file selected, returning...")
                return
            #source is a string with the path to the audio file
            #first check that it exists
            if not os.path.exists(source):
                print("File does not exist, returning...")
                return
            #load the file
            print("Loading file...")
            self.target_audio, self.file_sr = fileLoad(source)
            if self.target_audio == None or self.file_sr == None:
                return
            print("Starting separation...")
        else:
            #TODO: implement sync mode
            print("TODO!")
            #pick target
            #stgart audio pipeline

        self.hasTarget = True
        return True
    def start(self):
        if self.hasTarget:
            if self.mode == "async":
               self.asyncSeparate()
            else:
               self.separateStream()
        else:
            print("No target set, returning...")
            return

    def updateMode(self, mode):
        self.mode = mode
        #if there is anything running interrupt it
        #TODO
    def separate(self):
        #TODO
        self.separating=True
        while self.separating: #TODO
            chunk = self.getChunk(self)
            if chunk == None:
                self.separating = False
                break
            #separate chunk
            with torch.inference_mode():
                out = self.model(chunk)
            #TODO: choose a sound library before continuing
            #choices are sounddevice, soundcard, pyaudio, ...
            #chose https://pypi.org/project/sounddevice/ for better documentation
            #and explicitely mentioning our usage among use cases
        

            #either write directly or save in an array and output all at the end
            #maybe both can be done (e.g. saving mode if the user wants to save the separated
            #tracks - though this would be out of the use cases for the app)

    def getChunk(self):
        #async is "easy" mode
        if self.mode == "async":
            if self.async_index >= len(self.target_audio):
                return None
            chunk = self.target_audio[self.async_index:self.async_index+self.chunkLength,:]
            if self.win != None:
                chunk = chunk*self.win
            self.async_index += self.hop
            return chunk
        elif self.mode == "sync": #I have explicitely separated it from "else" just in case 
            #something weird happens with mode
            #TODO
            return None
        return None

    def hasNextChunk(self):
        #TODO
        return True
        return False
    

    def debug(self):
        print(self.async_index)

    def loadModel(self, modelType, debug=True):
        self.model = get_model(modelType, self.config)
        try:
            weights_path = self.config['global']['weights_path']
            load_weights(self.model, weights_path, useCuda=self.config['global']['device'].startswith('cuda'))
        except:
            self.async_index = 10

            print("Error loading weights -- check path and code")
            if debug:
                print("Script will continue with untrained net (debug=true).")
            else:
                sys.exit(1)
        return
    
def fileLoad(path):
    # i don't honestly know all of sf supported formats apart from the bigger ones
    #...hence:
    try:
        audio, sr = sf.read(path)
    except:
        print("Error reading file")
        return None, None
    return audio, sr