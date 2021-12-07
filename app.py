# -*- coding: utf-8 -*-
from tkinter import  Tk, filedialog, messagebox, Button, W, E, N, S, Label, Text, END
import os
import cv2
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
        
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        self.img = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                print(filename)
                self.img.append(os.path.join(folder,filename))
        self.checkButton['state'] = "normal"
        
        # try:
        #     file_path = filedialog.askopenfilename()
        #     img = Image.open(file_path)
        #     img.thumbnail((1000, 650),)
        #     photo = ImageTk.PhotoImage(img) 
        #     self.label.configure(image = photo, height=photo.height(), width=photo.width())
        #     self.label.image = photo
        #     self.img = file_path
        # except:
        #     messagebox.showwarning(title=None, message="File not valid!")
        # else:
        #     self.checkButton['state'] = "normal"
    def saveResult(self):
        #luu hinh so do
        folder = filedialog.askdirectory(title="Chọn thư mục để lưu kết quả")
        folder_Sodo = os.path.join(folder, "Sodo")
        os.makedirs(folder_Sodo)
        os.chdir(folder_Sodo)
        for image in self.image_sodo:
            filename = image.split('/')[len(image.split('/')) - 1]
            cv2.imwrite(filename,cv2.imread(image))
        #luu hinh ko la so do
        folder_NotSodo = os.path.join(folder, "NotSodo")
        os.makedirs(folder_NotSodo)
        os.chdir(folder_NotSodo)
        for image in self.image_notSodo:
            filename = image.split('/')[len(image.split('/')) - 1]
            cv2.imwrite(filename,cv2.imread(image))

    def checkSodo(self):
        self.image_sodo=[]
        self.image_notSodo=[]
        for image in self.img:
            isSodo=is_sodo(image)[0]
            if isSodo:
                self.image_sodo.append(image)
            else:
                self.image_notSodo.append(image)
        self.saveResult()
        # Display text
        # self.text.insert(END, text)
        # self.result['text'] = "Hình ảnh là sổ đỏ" if isSodo else "Hình ảnh không là sổ đỏ"
        
    
    def handler(self):
        for file in iglob('ocrproc_*'):
            os.remove(file) 
        self.master.destroy()

root = Tk()
myapp = App(root)
root.mainloop()