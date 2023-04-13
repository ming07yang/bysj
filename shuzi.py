# coding=gbk
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # ����һ����ǩ����ʾ��Ƶ
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))

        # ����Ƶ�ļ�
        self.cap = cv2.VideoCapture('video.mp4')
        # self.cap = cv2.VideoCapture(0)

        # ����һ����ʱ��
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # �Ժ���Ϊ��λ���ü�ʱ����ʱ����
        self.timer.timeout.connect(self.update_frame)  # ����ʱ����timeout�ź����ӵ�update_frame����

        # ������ʱ��
        self.timer.start()

    def update_frame(self):
        # ��ȡ��һ֡
        ret, frame = self.cap.read()

        if ret:
            # ��֡ת��ΪQtͼ���ʽ
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            # �ڱ�ǩ����ʾͼ��
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        else:
            # �����ȡ�����ֹͣ��ʱ��
            self.timer.stop()

    def closeEvent(self, event):
        # �ڹرմ���ʱ�ͷ���Ƶ�ļ�
        self.cap.release()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())

