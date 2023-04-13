# coding=gbk
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
import sys


class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # 创建一个标签来显示视频
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 640, 480))

        # 打开视频文件
        self.cap = cv2.VideoCapture('video.mp4')
        # self.cap = cv2.VideoCapture(0)

        # 创建一个计时器
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # 以毫秒为单位设置计时器的时间间隔
        self.timer.timeout.connect(self.update_frame)  # 将计时器的timeout信号连接到update_frame方法

        # 启动计时器
        self.timer.start()

    def update_frame(self):
        # 读取下一帧
        ret, frame = self.cap.read()

        if ret:
            # 将帧转换为Qt图像格式
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            # 在标签上显示图像
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        else:
            # 如果读取完毕则停止计时器
            self.timer.stop()

    def closeEvent(self, event):
        # 在关闭窗口时释放视频文件
        self.cap.release()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())

