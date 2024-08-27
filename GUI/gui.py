import os
import app
import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
from audio.separator import separate, stopSeparation, asyncSeparate
class GUI:
    def __init__(self, DIRECTOR):
        self.root = tk.Tk()
        #set the app name to VocalSplitter
        self.root.title('VocalSplitter')
        #set initial size to 400x400
        self.root.geometry('400x400')
        self.separating = False
        self.sepButton = tk.Button(self.root)
        self.filepath = None
        self.mode = "async"
        self.DIRECTOR = DIRECTOR
        self.hasFilePath = False
        self.hasAudioSource = False
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
                if self.mode=="async" and not self.hasFilePath:
                    print("No file selected, returning...")
                    return
                if self.mode=='sync' and not self.hasAudioSource:
                    print("No audio stream source selected, returning...")
                    return
                print("Starting separation...")
                if self.mode=="async":
                    self.DIRECTOR.setTarget(self.filepath)
                else:
                    self.DIRECTOR.setTarget()
                self.separating = True
                self.DIRECTOR.separate()
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
    
    #def showSelection(self,*args):
    #    self.mode_label.config(text=f"Selected: {self.mode.get()}")
    def updateMode(self, *args):
        self.DIRECTOR.updateMode(self.mode)
        
    def addElements(self):
        #self.label = tk.Label(self.root, text='Hello, World!')
        #self.label.pack()
        self.sepBtnText = tk.StringVar()
        self.sepBtnText.set('Separate!')
        self.sepButton = tk.Button(self.root, textvariable=self.sepBtnText, command=lambda: self.buttonClick('separate'))
        self.sepButton.pack()
        #make a button with a folder icon
        folder_icon = Image.open("assets/folder_icon.png")
        folder_icon = folder_icon.resize((32,32))
        folder_icon = ImageTk.PhotoImage(folder_icon)
        self.folderButton = tk.Button(self.root, image=folder_icon, command=lambda: self.buttonClick('folder'))   
        self.folderButton.image = folder_icon
        self.folderButton.pack(pady=10)
        self.pickedStreamSource = tk.StringVar()
        self.pickedStreamSource.set('No file selected')
        self.streamSourceLabel = tk.Label(self.root, textvariable=self.pickedStreamSource)
        self.streamSourceLabel.pack()
        self.mode = tk.StringVar(value="async")

        options = ["async", "sync"]

        # Create an OptionMenu (dropdown)
        self.mode_label = tk.Label(self.root, text="MODE:")
        self.mode_label.pack(pady=20)
        dropdown = tk.OptionMenu(self.root, self.mode, *options)
        dropdown.pack(pady=0)

        # Label to display the selected option
        #self.mode_label = tk.Label(self.root, text="Selected: None")
        #self.mode_label.pack(pady=20)

        # Update label whenever the selection changes
        self.mode.trace('w', self.updateMode)
        self.voxLabel = tk.Label(self.root, text="Vox Gain: 1.0")
        self.voxLabel.pack(pady=10)
        self.voxSlider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=300, command=lambda x: self.updateGainLabels("vox"))
        self.voxSlider.set(1.0)
        self.voxSlider.pack(pady=0)
        self.instrLabel = tk.Label(self.root, text="Instr Gain: 1.0")
        self.instrLabel.pack(pady=10)
        self.instrSlider = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, length=300, command=lambda x: self.updateGainLabels("instr"))
        self.instrSlider.set(1.0)
        self.instrSlider.pack(pady=0)
        
    
    def getFilePath(self):
        file_path = filedialog.askopenfilename(title="Select the source audio file", initialdir=os.getcwd(), filetypes=[("WAV files", ".wav"), ("MP3 files", ".mp3"), ("FLAC files",".flac"),("All files", ".*")]) #i honestly don't know
        #which filetypes will be supported later on by whatever method reads them - for now i'm only including these
        
        print(file_path)
        if file_path=='' or file_path==None or file_path==' ':
            print("No file selected, returning...")
            return
        self.pickedStreamSource.set(file_path)
        self.filepath = file_path
        self.hasFilePath = True


    def updateGainLabels(self, key):
        if key=='vox':
            self.DIRECTOR.vox_gain = self.voxSlider.get()
            self.voxLabel.config(text=f"Vox Gain: {self.voxSlider.get():.2f}")

        elif key=='instr':
            self.DIRECTOR.instr_gain = self.instrSlider.get()
            self.instrLabel.config(text=f"Instr Gain: {self.instrSlider.get():.2f}")

