# coding=gbk
import sys
import csv
import cv2
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
from ui_wind.test011 import Ui_MainWindow
from ui_child.Child import Ui_ChildWindow
# from detect.chinup_det_keypoint_unite_infer import device


class MyMainWindow(QMainWindow,Ui_MainWindow):  #
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        self.Flag = False

    def show_video(self,filename):
        self.cap = cv2.VideoCapture(filename)
        # self.cap = cv2.VideoCapture(0)

        ##创建一个计时器
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # 以毫秒设置计时器
        self.timer.timeout.connect(self.update_frame)  # 将计时器的timeout信号连接到update_frame方法

        # 启动计时器
        self.timer.start()

    def update_frame(self):
        # 读取下一帧

        ret, frame = self.cap.read()
        if ret:
            #将帧转换成Qt图像格式
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            #在标签上显示图片
            self.label_vedio.setPixmap(QtGui.QPixmap.fromImage(image))
        else:
            #如果读取完毕停止计时器
            self.timer.stop()

     #启动系统
    def slot_start_video(self):
        self.Flag = True
        # self.label_Tips.setText("*系统已启动")

        self.pushButton_start_style.setStyleSheet("QPushButton{\n"
                                                "border:2px solid  rgba(68, 206, 246,150);\n"
                                                "background-color: rgba(0, 224, 121, 255);\n"
                                                "border-radius:15px; \n"
                                                "color: rgb(0, 51, 113);}\n")
        print("已开启系统")

        self.statusBar().showMessage('已开启系统！',)
        self.show_video('video.mp4')

     #关闭系统
    def slot_pause(self):
        #关闭视频窗口
        self.pushButton_start_style.setStyleSheet("QPushButton{\n"
                                                    "background-color: rgba(141, 75, 187,150);\n"
                                                    "border-radius:20px; \n"
                                                    "color: rgb(0, 51, 113);}\n"
                                                    "QPushButton:hover{background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 224, 121, 255), stop:1 rgba(255, 255, 255, 255));border:2px solid  rgba(14, 184, 58,150);\n"
                                                    "}\n"
                                                    "QPushButton:pressed{padding-top:2px;\n"
                                                    "padding-left:2px;}")
        self.Flag = False
        self.statusBar().showMessage('系统已关闭！', )
        self.cap.release()

        print("释放资源")


    #显示数字
    def slot_bt_show(self):

        s = self.lineEdit_input.text()
        # s = chinUpCnt
        # print(s)

        self.lcdNumber_01.display(s)
        print(int(self.lcdNumber_01.value()))

    #重置数字
    def slot_restart(self):
        self.lcdNumber_01.display(00)
        print(self.lcdNumber_01.value())

    #开始测试
    def slot_start_test(self):
        self.Bridge = False
        if self.Flag == False:
            self.label_Tips.setText("*系统还未开启，请开启系统!")
        else:
            if self.lineEdit_Name.text() == "":
                self.label_infor_tips.setText("请输入姓名")
                # QMessageBox.information(self, "消息提醒", "请输入姓名", QMessageBox.Yes)  #弹窗
            else:
                self.Name = self.lineEdit_Name.text()
                if self.lineEdit_Proj.text() == "":
                    self.label_infor_tips.setText("请输入专业班级信息!")
                    # QMessageBox.information(self, "消息提醒", "请输入专业班级信息!", QMessageBox.Yes) #弹窗
                else:
                    self.Pro = self.lineEdit_Proj.text()
                    if self.lineEdit_Num.text() == "":
                        self.label_infor_tips.setText("请输入学号!")
                        # QMessageBox.information(self, "消息提醒", "请输入学号!", QMessageBox.Yes) #弹窗
                    else:
                        self.label_infor_tips.setText("")
                        self.Num = self.lineEdit_Num.text()

                        self.Bridge = True
                        print(self.Name,self.Pro,self.Num)

    #提交按钮 槽函数   保存信息
    def save_data(self):
        if self.Bridge == False:
            QMessageBox.information(self, "消息提醒", "请先点击开始测试后再提交保存信息!", QMessageBox.Yes) #弹窗
        else:
            self.Sco = int(self.lcdNumber_01.value())
            with open("data.csv", "a", encoding="utf-8-sig", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.Name,self.Pro,self.Num,self.Sco])

            self.Bridge = False

            #提交完数据后，清空框内数据方便下次填写
            self.lineEdit_input.setText("00")
            self.lineEdit_Name.setText("")
            self.lineEdit_Proj.setText("")
            self.lineEdit_Num.setText("")

            self.lcdNumber_01.display(00)

            print('数据已保存!!!')

    #数据查看
    def slot_child(self):
        # 获取桌面属性

        # 通过桌面的宽和高来比例位置显示

        self.one = MyChildWindow()
        self.one.test()
        desktop = QApplication.desktop()
        self.one.move(int(desktop.width() * 0.7), int(desktop.height() * 0.2))
        self.one.show()

    #加载数据
    def load_csv_data(self, filename,table_name):
        # 打开CSV文件
        with open(filename) as csvfile:
            # 读取CSV文件中的数据
            reader = csv.reader(csvfile)
            data = [row for row in reader]

        # 设置表格的行和列数
        table_name.setRowCount(len(data))
        table_name.setColumnCount(len(data[0]))

        # 将数据添加到表格中
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                table_name.setItem(i, j, QtWidgets.QTableWidgetItem(col))

class MyChildWindow(QMainWindow,Ui_ChildWindow):  #
    def __init__(self, parent=None):
        super(MyChildWindow, self).__init__(parent)
        self.setupUi(self)

    def test(self):
        self.shiyixia = MyMainWindow.load_csv_data(self,'data.csv',self.Data_Table)
        pass



if __name__=="__main__":
    app = QApplication(sys.argv)
    mainWindow = MyMainWindow()

    # ui = test011.Ui_MainWindow()
    # ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())