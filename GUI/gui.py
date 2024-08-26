import os
import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
from audio.separator import separate, stopSeparation, asyncSeparate
class GUI:
    def __init__(self):
        self.root = tk.Tk()
        #set the app name to VocalSplitter
        self.root.title('VocalSplitter')
        #set initial size to 400x400
        self.root.geometry('400x400')
        self.separating = False
        self.sepButton = tk.Button(self.root)
        self.filepath = None
        self.mode = "async"
    def run(self):
        self.addElements()
        self.root.mainloop()

    def buttonClick(self, cmd=None):
        #this is probably a stupid way of implementing buttons but this
        #is my first tkinter project :)
        #I also dont want to use match/case since it's only for relatively new python vers
        if cmd == None:
            print("Button was clicked with a null command. Returning")
            return
        if cmd == 'separate':
            if not self.separating:
                if self.mode=="async" and self.filepath==None:
                    print("No file selected, returning...")
                    return
                print("Starting separation...")
                if self.mode=="async":
                    asyncSeparate(self.filepath)
                else:
                    separate()
                self.separating = True
                self.sepBtnText.set('Stop')
                #TODO: show progress bar in async mode (just show 
                #chunk n/N)
                #TODO: in sync mode add a darker overlay and a x on the top right
                # to stop
            else:
                print("Stopping separation...")
                self.separating = False
                self.sepBtnText.set('Separate!')
                #TODO
                stopSeparation()
        elif cmd == 'folder':
            print("Opening folder dialog...")
            self.getFilePath()
        return
    
    def addElements(self):
        #self.label = tk.Label(self.root, text='Hello, World!')
        #self.label.pack()
        self.sepBtnText = tk.StringVar()
        self.sepBtnText.set('Separate!')
        self.sepButton = tk.Button(self.root, textvariable=self.sepBtnText, command=lambda: self.buttonClick('separate'))
        self.sepButton.pack()
        #make a button with a folder icon
        folder_icon = Image.open("assets/folder_icon.png")
        folder_icon = folder_icon.resize((32,32), Image.ANTIALIAS)
        folder_icon = ImageTk.PhotoImage(folder_icon)
        self.folderButton = tk.Button(self.root, image=folder_icon, command=lambda: self.buttonClick('folder'))   
        self.folderButton.image = folder_icon
        self.folderButton.pack(pady=10)
        self.pickedStreamSource = tk.StringVar()
        self.pickedStreamSource.set('No file selected')
        self.streamSourceLabel = tk.Label(self.root, textvariable=self.pickedStreamSource)
        self.streamSourceLabel.pack()

    def getFilePath(self):
        file_path = filedialog.askopenfilename(title="Select the source audio file", initialdir=os.getcwd(), filetypes=[("WAV files", ".wav"), ("MP3 files", ".mp3"), ("FLAC files",".flac"),("All files", ".*")]) #i honestly don't know
        #which filetypes will be supported later on by whatever method reads them - for now i'm only including these
        
        print(file_path)
        self.pickedStreamSource.set(file_path)
        self.filepath = file_path
        self.hasFilePath = True
