from GUI import gui
from audio.audio_director import AudioDirector

class App:
    def __init__(self):
        self.DIRECTOR = AudioDirector()
        self.DIRECTOR.setup(cfgpath='cfg/cfg.yaml')
        self.gui = gui.GUI(self.DIRECTOR)
        
    def start(self):
        self.gui.run()
        
if __name__=='__main__':
    app = App()
    print ('Starting app...')
    app.start()