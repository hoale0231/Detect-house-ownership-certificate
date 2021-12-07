# -*- coding: utf-8 -*-
from tkinter import  Tk, filedialog, messagebox, Button, W, E, N, S, Label, Text, END
import os
import cv2
from glob import iglob
from main import is_sodo
from PIL import Image, ImageTk
import threading

class App():
    def __init__(self, master):
        self.master = master
        self.master.title("Sổ đỏ")
        self.master.protocol("WM_DELETE_WINDOW", self.handler)
        self.createWidgets()
  
    def createWidgets(self):
        # Create a label to display the img
        self.label = Label(self.master, height=20, width=100)
        self.label.grid(row=0, column=0, rowspan=2, columnspan=2,sticky=W+E+N+S, padx=5, pady=5)
        
        # Create button select img
        self.selectImageButton = Button(self.master)
        self.selectImageButton = Button(self.master, width=20, padx=3, pady=3)
        self.selectImageButton["text"] = "Select Image"
        self.selectImageButton["command"] = self.selectImage
        self.selectImageButton.grid(row=2, column=0, padx=2, pady=2)
        
        # Create button select img
        self.selectFolderButton = Button(self.master)
        self.selectFolderButton = Button(self.master, width=20, padx=3, pady=3)
        self.selectFolderButton["text"] = "Select Folder"
        self.selectFolderButton["command"] = self.selectFolder
        self.selectFolderButton.grid(row=2, column=1, padx=2, pady=2)
        
        # Create button check img
        self.checkButton = Button(self.master, width=20, padx=3, pady=3)
        self.checkButton["text"] = "Start"
        self.checkButton["command"] = self.start
        self.checkButton.grid(row=2, column=2, padx=2, pady=2)
        self.checkButton['state'] = "disable"
        
        # Create text widget
        self.text = Text(self.master, width=20, padx=2, pady=2)
        self.text.grid(row=0, column=2, sticky=W+E+N+S)
        
        self.result = Label(self.master)
        self.result.grid(row=1, column=2, padx=2, pady=2, sticky=W+E+N+S)
        
        # Mode: check image 0 or filter folder 1
        self.mode = 0
        # Save result for mode 1
        self.image_sodo=[]
        self.image_notSodo=[]
        
    def selectImage(self):
        self.mode = 0
        self.text.delete(1.0, END)
        self.result['text'] = ""
        try:
            self.setImg(filedialog.askopenfilename(title="Chọn ảnh"))
        except:
            messagebox.showwarning(title=None, message="File not valid!")
        else:
            self.checkButton['state'] = "normal"
    
    def setImg(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((1000, 650),)
        img = ImageTk.PhotoImage(img) 
        self.label.configure(image = img, height=img.height(), width=img.width())
        self.label.image = img
        self.img = file_path
         
    def selectFolder(self):
        self.mode = 1
        self.text.delete(1.0, END)
        self.result['text'] = ""
        
        folder = filedialog.askdirectory(title="Chọn thư mục chứa ảnh")
        self.imgs = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                self.imgs.append(os.path.join(folder,filename))
        self.checkButton['state'] = "normal"        
            
    def saveResult(self):
        #luu hinh so do
        owd = os.getcwd()
        folder = filedialog.askdirectory(title="Chọn thư mục để lưu kết quả")
        folder_Sodo = os.path.join(folder, "Sodo")
        if not os.path.exists(folder_Sodo):
            os.makedirs(folder_Sodo)
        os.chdir(folder_Sodo)
        
        for image in self.image_sodo:
            filename = image.split('\\')[-1]
            cv2.imwrite(filename, cv2.imread(image))
        #luu hinh ko la so do
        folder_NotSodo = os.path.join(folder, "NotSodo")
        if not os.path.exists(folder_NotSodo):
            os.makedirs(folder_NotSodo)
        os.chdir(folder_NotSodo)
        for image in self.image_notSodo:
            filename = image.split('\\')[-1]
            cv2.imwrite(filename, cv2.imread(image))
            
        os.chdir(owd)

    def start(self):
        if self.mode == 0:
            threading.Thread(target=self.checkImg, args=[self.img]).start()
        else:
            threading.Thread(target=self.filterFolder).start()
            
        self.result['text'] = "Loading..."
    
    def filterFolder(self):
        self.image_sodo=[]
        self.image_notSodo=[]
        threads = []
        for image in self.imgs:
            self.checkImg(image)
            # self.setImg(image)
        #     t = threading.Thread(target=self.checkImg, args=[image])
        #     t.start()
        #     threads.append(t)
        # for thread in threads:
        #     thread.join()
        self.saveResult()       
        self.result['text'] = "Hoàn thành"
         
    
    def checkImg(self, img):
        isSodo, text = is_sodo(img)
        if self.mode == 1:
            if isSodo:
                self.image_sodo.append(img)
            else:
                self.image_notSodo.append(img)
        else:
            self.text.insert(END, text)
            self.result['text'] = "Hình ảnh là sổ đỏ" if isSodo else "Hình ảnh không là sổ đỏ"

    def handler(self):
        for file in iglob('ocrproc_*'):
            os.remove(file) 
        self.master.destroy()

root = Tk()
myapp = App(root)
root.mainloop()