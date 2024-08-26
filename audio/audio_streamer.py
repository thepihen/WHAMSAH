import os
import numpy as np
import soundfile as sf
class AudioDirector:
    def __init__(self, mode="async"):
        self.mode = mode
        self.started=False
    def setup(self):
        #load model
        #TODO
        #get internal sample rate
        #if != 44100, resample to 44100 to feed to the model half second inputs
        #TODO
        pass
    def start(self, source=None, target_sr=None):
        self.setup()
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

        self.started = True
        return True


def fileLoad(path):
    # i don't honestly know all of sf supported formats apart from the bigger ones
    #...hence:
    try:
        audio, sr = sf.read(path)
    except:
        print("Error reading file")
        return None, None
    return audio, sr