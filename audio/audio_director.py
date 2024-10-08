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
        self.syncStream = None
        self.inputDevID = None
        self.outputDevID = None
        self.win = None
        self.chunk = None
        self.half_chunk = None
        self.sync_half_chunk = None
        self.sync_chunk = None

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
                self.config = yaml.load(stream,Loader=yaml.FullLoader)
                self.chunkLength = self.config['separator']['chunk']
                self.hop = self.config['separator']['hop']
                self.blocksz = int(self.chunkLength / (self.chunkLength / self.hop))
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
                if self.win is not None:
                    #duplicate the windows so that it is a [winL, 2] array
                    self.win = np.stack((self.win, self.win), axis=1)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(1)
        
        self.stream = sd.OutputStream(samplerate=self.sr, blocksize=self.blocksz, callback=self.streamCallback, finished_callback=event.set)
        
        self.loadModel(self.config['global']['model'])
            
    def setTarget(self, source=None, target_sr=None):
        print("Setting target...")
        
        self.async_index = 0 #just in case
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
        if mode != self.mode:
            if self.mode == "async":
                if self.stream is not None:
                    self.stream.abort()
                    self.stream = None
            else:
                if self.syncStream is not None:
                    self.syncStream.abort()
                    self.syncStream = None
                    self.sync_half_chunk = None
        self.half_chunk = None
        self.mode = mode
        
        #if there is anything running interrupt it

    def separate(self):
        self.separating=True
        if self.mode == "async":
            if self.stream is None:
                self.stream = sd.OutputStream(blocksize=self.blocksz, callback=self.streamCallback, finished_callback=event.set)
                print("Reassigned stream")
            self.stream.start()
        elif self.mode == "sync":
            #setup an input stream from self.inputDevice
            #setup an output stream from self.outputDevice
            #start both
            if self.syncStream == None:
                self.syncStream = sd.Stream(blocksize=self.blocksz, callback=self.syncStreamCallback, finished_callback=event.set, device=[self.inputDevID, self.outputDevID])
                print("Reassigned syncStream")
            self.syncStream.start()
        
    def getChunk(self):
        #async is "easy" mode
        if self.mode == "async":
            if self.async_index >= len(self.target_audio):
                return None
            try:
                chunk = self.target_audio[self.async_index:self.async_index+self.chunkLength,:]
            except:
                chunk = self.target_audio[self.async_index:,:]
                #zero pad
                chunk = np.pad(chunk, ((0, self.chunkLength - chunk.shape[0]), (0,0)), 'constant')
            #if self.win is not None:
            #    chunk = chunk*self.win
            self.async_index += self.hop
            return chunk
        elif self.mode == "sync": #I have explicitely separated it from "else" just in case 
            #something weird happens with mode
            return self.chunk
        return None

    def hasNextChunk(self):
        #TODO this function is not used for now
        return True
    
    def streamCallback(self,outdata, frames, time, status=True):
        print(frames)
        if status:
            print(status, file=sys.stderr)
        chunk = self.getChunk()
        if chunk is None and self.mode=="async":
            self.stopSeparation()
            return
        if chunk is None and self.mode=="sync":
            return
        #separate chunk
        #chunk to torch
        chunk = torch.from_numpy(chunk).to(self.config['global']['device'])
        chunk = chunk.T
        #chunk should be float32
        chunk = chunk.float()
        chunk = chunk.unsqueeze(0)
        #outdata[:] = np.transpose(torch.squeeze(chunk, dim=0).to("cpu").numpy()) #if you want to test a pass-through
        
        with torch.inference_mode():
            out = self.model(chunk)
        #chose https://pypi.org/project/sounddevice/ for better documentation
        #and explicitely mentioning our usage among use cases
        #convert to float32
        out = np.transpose(torch.squeeze(out, dim=0).to("cpu").numpy())#[:frames]
        chunk = np.transpose(torch.squeeze(chunk, dim=0).to("cpu").numpy())#[:frames]

        #multiply both out and chunk by win
        if self.win is not None:
            out = out*self.win

        if self.half_chunk is not None:
            out[:self.blocksz] += self.half_chunk
        if self.blocksz != self.chunkLength:
            self.half_chunk = out[frames:]
        out = out[:frames]
        instr_part = chunk[:frames] - out

        #self.outChunk = out
        #self.outChunk = np.transpose(out)
        #print(self.outChunk.shape)
        #assert 0
        #outdata[:] = self.outChunk[:frames]
        outdata[:] = instr_part * self.instr_gain + out * self.vox_gain #np.transpose(out)[:frames]
    
    def syncStreamCallback(self, indata, outdata, frames, time, status):
        #TODO
        print("got input chunk")
        #print(frames)
        #print(indata.shape)
        if indata.shape[1] > 2:
            indata = indata[:,:2]
        elif indata.shape[1] < 2:
            indata = np.pad(indata, ((0,0), (0,2-indata.shape[1])), 'constant')
        #we absolutely want stereo data for the model
        if indata is None and self.mode=="sync":
            return
        
        if self.sync_half_chunk is None:
            self.sync_half_chunk = indata
            outdata[:] = np.zeros((frames, 2))
            return
        else:
            inp = np.concatenate((self.sync_half_chunk, indata), axis=0)
            self.sync_half_chunk = inp[self.blocksz:] #we're basically simulating a N/2 hop

        #we assume that the data is half the chunk length
        if inp.shape[0] > self.chunkLength:
            #sorry not sorry
            inp = inp[:self.chunkLength]

        chunk = torch.from_numpy(inp).to(self.config['global']['device'])
        chunk = chunk.T
        #chunk should be float32
        chunk = chunk.float()
        chunk = chunk.unsqueeze(0)
        #outdata[:] = np.transpose(torch.squeeze(chunk, dim=0).to("cpu").numpy()) #if you want to test a pass-through
        
        with torch.inference_mode():
            out = self.model(chunk)
        #chose https://pypi.org/project/sounddevice/ for better documentation
        #and explicitely mentioning our usage among use cases
        #convert to float32
        out = np.transpose(torch.squeeze(out, dim=0).to("cpu").numpy())#[:frames]
        chunk = np.transpose(torch.squeeze(chunk, dim=0).to("cpu").numpy())#[:frames]
        #instr_part = chunk - out
        if self.win is not None:
            out = out*self.win

        if self.half_chunk is not None:
            out[:self.blocksz] += self.half_chunk

        if self.blocksz != self.chunkLength:
            self.half_chunk = out[frames:]
            
        out = out[:frames]
        instr_part = chunk[:frames] - out
        #self.outChunk = out
        #self.outChunk = np.transpose(out)
        #print(self.outChunk.shape)
        #assert 0
        #outdata[:] = self.outChunk[:frames]
        outdata[:] = instr_part * self.instr_gain + out * self.vox_gain

    def syncInStreamCallback(self, indata, frames, time, status):
        if indata.shape[1] > 2:
            indata = indata[:,:2]
        elif indata.shape[1] < 2:
            indata = np.pad(indata, ((0,0), (0,2-indata.shape[1])), 'constant')
        #we absolutely want stereo data for the model
        if indata is None and self.mode=="sync":
            return
        self.sync_chunk = indata
        self.sync_half_chunk = self.sync_chunk[self.blocksz:]

    def syncOutStreamCallback(self, outdata, frames, time, status):
        try:
            chunk = torch.from_numpy(self.sync_chunk).to(self.config['global']['device'])
        except:
            return
        chunk = chunk.T
        #chunk should be float32
        chunk = chunk.float()
        chunk = chunk.unsqueeze(0)
        with torch.inference_mode():
            out = self.model(chunk)
        #chose https://pypi.org/project/sounddevice/ for better documentation
        #and explicitely mentioning our usage among use cases
        #convert to float32
        out = np.transpose(torch.squeeze(out, dim=0).to("cpu").numpy())#[:frames]
        chunk = np.transpose(torch.squeeze(chunk, dim=0).to("cpu").numpy())#[:frames]
        #instr_part = chunk - out
        if self.win is not None:
            out = out*self.win

        if self.half_chunk is not None:
            out[:self.blocksz] += self.half_chunk

        if self.blocksz != self.chunkLength:
            self.half_chunk = out[frames:]
            
        out = out[:frames]
        instr_part = chunk[:frames] - out
        #self.outChunk = out
        #self.outChunk = np.transpose(out)
        #print(self.outChunk.shape)
        #assert 0
        #outdata[:] = self.outChunk[:frames]
        outdata[:] = instr_part * self.instr_gain + out * self.vox_gain
        self.sync_chunk = None

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
        self.model = self.model.to(self.config['global']['device'])
        return
    
    def stopSeparation(self):
        self.separating = False
        #self.stream.stop()
        if self.mode=="async":
            self.stream.stop()
        else:
            self.syncStream.stop()
        #self.stream = None
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
        self.inputDevID = device['index']
        print(f"AD: Input device updated to {device['name']}")
        print(device)
    
    def updateOutputDevice(self, device):
        #TODO
        self.outputDevice = device
        self.outputDevID = device['index']
        print(f"AD: Output device updated to {device['name']}")
        #print(device)

def fileLoad(path):
    # i don't honestly know all of sf supported formats apart from the bigger ones
    #...hence:
    try:
        audio, sr = sf.read(path)
    except:
        print("Error reading file")
        return None, None
    return audio, sr