import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
import torchvision.transforms as transforms
from model.squeezenet import SqueezeNet
import numpy as np
from PIL import Image
from torchvision import models
from PIL import Image
import torch.nn as nn
def check_and_convert_to_rgb(img):
    # Check if image is in RGB format, if not convert it
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img
transform1 = transforms.Compose([                         
                                 transforms.ToTensor(),                               
                                 transforms.Resize((32, 32)),
                                 transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
                                 
])
model_path = "checkpoints/model_8.ckpt"
model=models.squeezenet1_0(pretrained=True)
model.load_state_dict(torch.load("squeezenet1_0-a815701f.pth"))
model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=31,
                                kernel_size=1) # #Change the output of the last layer of the network to 20 categories
model.num_classes = 31 #Change the number of classification categories of the network

model.load_state_dict(torch.load(model_path)) # read ckpt file 
class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Image Recognition'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 300
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setGeometry(50, 50, 300, 200)

        self.btn1 = QPushButton('Select Image', self)
        self.btn1.move(50, 260)
        self.btn1.clicked.connect(self.openFileNameDialog)

        self.btn2 = QPushButton('Predict', self)
        self.btn2.move(250, 260)
        self.btn2.clicked.connect(self.predict)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","Image Files (*.bmp *.png *.jpg *.jpeg);;All Files (*)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(300, 200, Qt.KeepAspectRatio))

    def predict(self):
        img = Image.open(self.image_path)
        img = check_and_convert_to_rgb(img)
        img = transform1(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            model.eval()
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            class_dict = predicted.item()
        class_dict = {0: "010", 1: "011", 2: "012",
              3:"013",4:"014",5:"015",
               6:"016",7:"017",8:"018",
                9:"019",10:"020",11:"DU",
                12:"DUY",13:"KHANG",14:"HAN",
                15:"HIEU",16:"KHOA",17:"NHAN",
                18:"NHI",19:"QUAN",20:"QUANG",
                21:"T_QUAN",22:"TAI",23:"TAN",
                24:"THAI",25:"THONG",26:"TIN",
                27:"TRINH",28:"VIET",29:"VINH",
                30:"VY"              
              
              }

        print("Predicted class:", class_dict[predicted.item()])

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
