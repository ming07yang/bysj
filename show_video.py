# coding=gbk
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys

class Show_Video():
    def __int__(self):
        super().__int__()

    def show_video(self):
        self.cap = cv2.VideoCapture('video.mp4')

        ##����һ����ʱ��
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50) #�Ժ������ü�ʱ��
        self.timer.timeout.connect(self.update_frame)# ����ʱ����timeout�ź����ӵ�update_frame����

        #������ʱ��
        self.timer.start()

    def update_frame(self):
        #��ȡ��һ֡
        ret,frame = self.cap.read()


