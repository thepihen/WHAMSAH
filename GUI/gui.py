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
        self.root.geometry('600x600')
        self.separating = False
        self.sepButton = tk.Button(self.root)
        self.filepath = None
        self.mode = "async"
        self.DIRECTOR = DIRECTOR
        self.hasFilePath = False
        self.hasAudioSource = False
        self.devices = self.DIRECTOR.getAvailableDevices()
        self.sortDevices() #classify devices as input or output
        #self.inputDevices and self.outputDevices are now available
        self.APIs = self.DIRECTOR.getHostAPIs() #each API has a name and a list of devices
        self.APINames = [api['name'] for api in self.APIs]
        self.availableInputs = ["-"]
        self.availableOutputs = ["-"]
        #that are available for it under 'devices'
    def run(self):
        self.addElements()
        self.root.mainloop()

    def buttonClick(self, cmd=None):
        #this is probably a stupid way of implementing buttons but this
        #is my first tkinter project :)
        #I also dont want to use match/case since it's only for relatively new python vers
        print(f"Button was clicked with command: {cmd}")
        print(f"Mode: {self.mode}, filepath: {self.filepath}, hasFilePath: {self.hasFilePath}, hasAudioSource: {self.hasAudioSource}")
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
                    print("*/****")
                    print(self.filepath)
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
                self.sepBtnText.set('Play/Separate!')
                #TODO
                self.DIRECTOR.stopSeparation()
        elif cmd == 'folder':
            print("Opening folder dialog...")
            self.getFilePath()
        return
    
    #def showSelection(self,*args):
    #    self.mode_label.config(text=f"Selected: {self.mode.get()}")
    def updateMode(self, *args):
        self.mode = self.modeLabelVar.get()
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
        self.modeLabelVar = tk.StringVar(value="async")

        options = ["async", "sync"]

        # Create an OptionMenu (dropdown)
        self.mode_label = tk.Label(self.root, text="MODE:")
        self.mode_label.pack(pady=20)
        dropdown = tk.OptionMenu(self.root, self.modeLabelVar, *options)
        dropdown.pack(pady=0)

        # Label to display the selected option
        #self.mode_label = tk.Label(self.root, text="Selected: None")
        #self.mode_label.pack(pady=20)

        # Update label whenever the selection changes
        self.modeLabelVar.trace('w', self.updateMode)
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
        
        #API SELECTIOn
        self.apiLabel = tk.Label(self.root, text="HOST API:")
        self.apiLabel.pack(pady=20)
        self.apiLabelVar = tk.StringVar(value="None")
        APIdropdown = tk.OptionMenu(self.root, self.apiLabelVar, *self.APINames)
        self.apiLabelVar.trace('w', self.updateAPI)
        APIdropdown.pack(pady=0)

        #INPUT SELECTION
        self.input_label = tk.Label(self.root, text="INPUT DEVICE:")
        self.inputLabelVar = tk.StringVar(value="None")
        self.input_label.pack(pady=20)
        self.INPUTdropdown = tk.OptionMenu(self.root, self.inputLabelVar, *self.availableInputs)
        self.INPUTdropdown.pack(pady=0)
        self.inputLabelVar.trace('w', self.updateInput)
        
        #repeat for output devices making the new dropdown appear alongside the input one
        self.output_label = tk.Label(self.root, text="OUTPUT DEVICE:")
        self.outputLabelVar = tk.StringVar(value="None")
        self.output_label.pack(pady=20)
        self.OUTPUTdropdown = tk.OptionMenu(self.root, self.outputLabelVar, *self.availableOutputs)
        self.OUTPUTdropdown.pack(pady=0)
        self.outputLabelVar.trace('w', self.updateOutput)



    
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


    def sortDevices(self):
        #classify devices in self.devices as input or output devices
        self.inputDevices = []
        self.outputDevices = []
        for device in self.devices:
            if device['max_input_channels']>0:
                self.inputDevices.append(device)
            if device['max_output_channels']>0:
                self.outputDevices.append(device)

    def updateAPI(self, *args):
        #update the available inputs and outputs based on the selected API
        selectedAPI = self.apiLabelVar.get()
        print(f"Selected API: {selectedAPI}")
        #print(self.inputDevices)
        #assert 0
        for api in self.APIs:
            if api['name'] == selectedAPI:
                self.devicesIndexes = api['devices']
                self.availableInputs = [device['name'] for device in self.inputDevices if device['index'] in self.devicesIndexes]
                self.availableOutputs = [device['name'] for device in self.outputDevices if device['index'] in self.devicesIndexes]
                print(f"Available inputs: {self.availableInputs}")
                print(f"Available outputs: {self.availableOutputs}")
                break
        #update the dropdowns
        self.updateDropdowns()

    def updateDropdowns(self):
        #update the dropdowns based on the available inputs and outputs
        #self.INPUTdropdown = tk.OptionMenu(self.root, self.inputLabelVar, *self.availableInputs)
        #self.
        menu = self.INPUTdropdown["menu"]
        menu.delete(0, "end")
        for option in self.availableInputs:
            menu.add_command(label=option, command=lambda value=option: self.inputLabelVar.set(value))
        self.inputLabelVar.set(self.availableInputs[0])

        menuout = self.OUTPUTdropdown["menu"]
        menuout.delete(0, "end")
        for option in self.availableOutputs:
            menuout.add_command(label=option, command=lambda value=option: self.outputLabelVar.set(value))
        self.outputLabelVar.set(self.availableOutputs[0])
        #self.outputLabelVar.set(self.availableOutputs[0])
    
    def updateInput(self, *args):
        #update the input device based on the selected input device
        selectedInput = self.inputLabelVar.get()
        print(f"Selected input: {selectedInput}")
        for device in self.inputDevices:
            if device['name'] == selectedInput:
                self.DIRECTOR.updateInputDevice(device)
                break
    def updateOutput(self, *args):
        #update the output device based on the selected output device
        selectedOutput = self.outputLabelVar.get()
        print(f"Selected output: {selectedOutput}")
        for device in self.outputDevices:
            if device['name'] == selectedOutput:
                self.DIRECTOR.updateOutputDevice(device)
                break