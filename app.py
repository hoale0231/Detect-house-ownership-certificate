# -*- coding: utf-8 -*-
from tkinter import  Tk, filedialog, messagebox, Button, W, E, N, S, Label, Text, END
from os import remove
from glob import iglob
from main import is_sodo
from PIL import Image, ImageTk

class App():
    def __init__(self, master):
        self.master = master
        self.master.title("Sổ đỏ")
        self.master.protocol("WM_DELETE_WINDOW", self.handler)
        self.createWidgets()
  
    def createWidgets(self):
        # Create a label to display the img
        self.label = Label(self.master, height=20, width=100)
        self.label.grid(row=0, column=0, rowspan=2, sticky=W+E+N+S, padx=5, pady=5)
        
        # Create button select img
        self.selectImageButton = Button(self.master)
        self.selectImageButton = Button(self.master, width=20, padx=3, pady=3)
        self.selectImageButton["text"] = "Select Image"
        self.selectImageButton["command"] = self.selectImage
        self.selectImageButton.grid(row=2, column=0, padx=2, pady=2)
        
        # Create button check img
        self.checkButton = Button(self.master, width=20, padx=3, pady=3)
        self.checkButton["text"] = "Check"
        self.checkButton["command"] = self.checkSodo
        self.checkButton.grid(row=2, column=1, padx=2, pady=2)
        self.checkButton['state'] = "disable"
        
        # Create text widget
        self.text = Text(self.master, width=20, padx=2, pady=2)
        self.text.grid(row=0, column=1, sticky=W+E+N+S)
        
        self.result = Label(self.master)
        self.result.grid(row=1, column=1, padx=2, pady=2, sticky=W+E+N+S)
        
    def selectImage(self):
        self.text.delete(1.0, END)
        self.result['text'] = ""
        try:
            file_path = filedialog.askopenfilename()
            img = Image.open(file_path)
            img.thumbnail((1000, 650),)
            photo = ImageTk.PhotoImage(img) 
            self.label.configure(image = photo, height=photo.height(), width=photo.width())
            self.label.image = photo
            self.img = file_path
        except:
            messagebox.showwarning(title=None, message="File not valid!")
        else:
            self.checkButton['state'] = "normal"
    
    def checkSodo(self):
        isSodo, text =is_sodo(self.img)
        # Display text
        self.text.insert(END, text)
        self.result['text'] = "Hình ảnh là sổ đỏ" if isSodo else "Hình ảnh không là sổ đỏ"
        
    
    def handler(self):
        for file in iglob('ocrproc_*'):
            remove(file) 
        self.master.destroy()

root = Tk()
myapp = App(root)
root.mainloop()