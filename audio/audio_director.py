import os
import sys
import librosa
import threading
import numpy as np
import soundfile as sf
import torch
import yaml
from utils import get_model, load_weights
import sounddevice as sd
#in tests sd consistently reported the wrong samplerate for my sound card 
#(Focusrite Scarlett 2i2 3rd gen)...
#in the end it still works but keep this in mind

event = threading.Event()

class AudioDirector:
    def __init__(self, mode="async"):
        self.mode = mode
        self.started=False
        self.async_index = 0
        self.last_as_idx = 0
        self.async_rep_index = 0
        self.vox_gain = 1.0
        self.instr_gain = 1.0
        self.stream = None 
        self.win = None

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
                self.sr = self.config['separator']['sr']
                #match winType with np.winType
                if winType == "hanning" or winType =='hann':
                    self.win = np.hanning(self.chunkLength)
                elif winType == "hamming":
                    self.win = np.hamming(self.chunkLength)
                elif winType == "blackman":
                    self.win = np.blackman(self.chunkLength)
                elif winType == "bartlett": 
                    self.win = np.bartlett(self.chunkLength)
                else:
                    print("Invalid or no window type - using no window")
                    self.win = None
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
        
        self.stream = sd.OutputStream(samplerate=self.sr, blocksize=self.chunkLength, callback=self.streamCallback, finished_callback=event.set)
        
        self.loadModel(self.config['global']['model'])
            
    def setTarget(self, source=None, target_sr=None):
        print("Setting target...")
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
            self.file_sr = None
            self.target_audio, self.file_sr = fileLoad(source)
            if self.file_sr == None:
                return
            if self.sr != self.file_sr:
                print("Sample rate mismatch, modifying data sr...")
                #resample the audio
                self.target_audio = librosa.resample(self.target_audio.T, orig_sr=self.file_sr, target_sr=self.sr).T
                
                
            print("AD: Starting separation...")
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
        print(f"AD: Updating mode to {mode}")
        self.mode = mode
        #if there is anything running interrupt it
        #TODO
    def separate(self):
        #TODO
        self.separating=True
        if self.stream is None:
            self.stream = sd.OutputStream(blocksize=65536, callback=self.streamCallback, finished_callback=event.set)
        self.stream.start()
        with self.stream:
            #pass
            input() #TODO: find a better way to do this
            #Right now, stopping and restarting the stream will break the program
            #and it will have to be restarted
            #It is acceptable only because it's a proof of concept
        """
        while self.separating: #TODO
            print(self.async_index)
            chunk = self.getChunk()
            if chunk is None:
                self.separating = False
                break
            #separate chunk
            with torch.inference_mode():
                out = self.model(chunk)
            #chose https://pypi.org/project/sounddevice/ for better documentation
            #and explicitely mentioning our usage among use cases

            #convert to float32
            self.outChunk = out.astype(np.float32)
            with self.stream:
                input()

            #either write directly or save in an array and output all at the end
            #maybe both can be done (e.g. saving mode if the user wants to save the separated
            #tracks - though this would be out of the use cases for the app)
        """
    def getChunk(self):
        #async is "easy" mode
        if self.mode == "async":
            if self.async_index >= len(self.target_audio):
                return None
            chunk = self.target_audio[self.async_index:self.async_index+self.chunkLength,:]
            if self.win is not None:
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
    
    def streamCallback(self,outdata, frames, time, status=True):
        if status:
            print(status, file=sys.stderr)
        chunk = self.getChunk()
        if chunk is None:
            self.separating = False
            return
        #separate chunk
        with torch.inference_mode():
            out = self.model(chunk)
        #chose https://pypi.org/project/sounddevice/ for better documentation
        #and explicitely mentioning our usage among use cases
        #convert to float32
        self.outChunk = out.astype(np.float32)
        outdata[:] = self.outChunk[:frames]





    def debug(self):
        print(self.async_index)

    def loadModel(self, modelType, debug=True):
        self.model = get_model(modelType, self.config)
        try:
            weights_path = self.config['global']['weights_path']
            load_weights(self.model, weights_path, useCuda=self.config['global']['device'].startswith('cuda'))
        except:
            print("Error loading weights -- check path and code")
            if debug:
                print("Script will continue with untrained net (debug=true).")
            else:
                sys.exit(1)
        return
    
    def stopSeparation(self):
        self.separating = False
        self.stream.stop()
        self.stream.close()
        self.stream = None
        print("AD: SEPARATION STOPPED")
        #if async, and if the choice was to save - export the output
        #TODO
        return
    
    def getAvailableDevices(self, device=None, kind=None):
        """
        device (int or str, optional) – Numeric device ID or device name substring(s). If specified, information about only the given device is returned in a single dictionary.
        kind ({‘input’, ‘output’}, optional) – If device is not specified and kind is 'input' or 'output', a single dictionary is returned with information about the default input or output device, respectively.
        """
        return sd.query_devices(device=device, kind=kind)
    def getHostAPIs(self, index=None):
        return sd.query_hostapis(index=index)
    


    def updateInputDevice(self, device):
        #TODO
        self.inputDevice = device
        print(f"AD: Input device updated to {device}")
    
    def updateOutputDevice(self, device):
        #TODO
        self.outputDevice = device
        print(f"AD: Output device updated to {device}")

def fileLoad(path):
    # i don't honestly know all of sf supported formats apart from the bigger ones
    #...hence:
    try:
        audio, sr = sf.read(path)
    except:
        print("Error reading file")
        return None, None
    return audio, sr