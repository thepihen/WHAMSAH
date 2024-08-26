from GUI import gui

class App:
    def __init__(self):
        self.gui = gui.GUI()
    def start(self):
        self.gui.run()

if __name__=='__main__':
    app = App()
    print ('Starting app...')
    app.start()