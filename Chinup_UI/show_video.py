# coding=gbk
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Show_Video():
    def __int__(self):
        super().__int__()

    def show_video(self):
        self.cap = cv2.VideoCapture('video.mp4')

        ##创建一个计时器
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50) #以毫秒设置计时器
        self.timer.timeout.connect(self.update_frame)# 将计时器的timeout信号连接到update_frame方法

        #启动计时器
        self.timer.start()

    def update_frame(self):
        #读取下一帧
        ret,frame = self.cap.read()


