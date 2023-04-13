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

        ##����һ����ʱ��
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)  # �Ժ������ü�ʱ��
        self.timer.timeout.connect(self.update_frame)  # ����ʱ����timeout�ź����ӵ�update_frame����

        # ������ʱ��
        self.timer.start()

    def update_frame(self):
        # ��ȡ��һ֡

        ret, frame = self.cap.read()
        if ret:
            #��֡ת����Qtͼ���ʽ
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            #�ڱ�ǩ����ʾͼƬ
            self.label_vedio.setPixmap(QtGui.QPixmap.fromImage(image))
        else:
            #�����ȡ���ֹͣ��ʱ��
            self.timer.stop()

     #����ϵͳ
    def slot_start_video(self):
        self.Flag = True
        # self.label_Tips.setText("*ϵͳ������")

        self.pushButton_start_style.setStyleSheet("QPushButton{\n"
                                                "border:2px solid  rgba(68, 206, 246,150);\n"
                                                "background-color: rgba(0, 224, 121, 255);\n"
                                                "border-radius:15px; \n"
                                                "color: rgb(0, 51, 113);}\n")
        print("�ѿ���ϵͳ")

        self.statusBar().showMessage('�ѿ���ϵͳ��',)
        self.show_video('video.mp4')

     #�ر�ϵͳ
    def slot_pause(self):
        #�ر���Ƶ����
        self.pushButton_start_style.setStyleSheet("QPushButton{\n"
                                                    "background-color: rgba(141, 75, 187,150);\n"
                                                    "border-radius:20px; \n"
                                                    "color: rgb(0, 51, 113);}\n"
                                                    "QPushButton:hover{background-color:qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(0, 224, 121, 255), stop:1 rgba(255, 255, 255, 255));border:2px solid  rgba(14, 184, 58,150);\n"
                                                    "}\n"
                                                    "QPushButton:pressed{padding-top:2px;\n"
                                                    "padding-left:2px;}")
        self.Flag = False
        self.statusBar().showMessage('ϵͳ�ѹرգ�', )
        self.cap.release()

        print("�ͷ���Դ")


    #��ʾ����
    def slot_bt_show(self):

        s = self.lineEdit_input.text()
        # s = chinUpCnt
        # print(s)

        self.lcdNumber_01.display(s)
        print(int(self.lcdNumber_01.value()))

    #��������
    def slot_restart(self):
        self.lcdNumber_01.display(00)
        print(self.lcdNumber_01.value())

    #��ʼ����
    def slot_start_test(self):
        self.Bridge = False
        if self.Flag == False:
            self.label_Tips.setText("*ϵͳ��δ�������뿪��ϵͳ!")
        else:
            if self.lineEdit_Name.text() == "":
                self.label_infor_tips.setText("����������")
                # QMessageBox.information(self, "��Ϣ����", "����������", QMessageBox.Yes)  #����
            else:
                self.Name = self.lineEdit_Name.text()
                if self.lineEdit_Proj.text() == "":
                    self.label_infor_tips.setText("������רҵ�༶��Ϣ!")
                    # QMessageBox.information(self, "��Ϣ����", "������רҵ�༶��Ϣ!", QMessageBox.Yes) #����
                else:
                    self.Pro = self.lineEdit_Proj.text()
                    if self.lineEdit_Num.text() == "":
                        self.label_infor_tips.setText("������ѧ��!")
                        # QMessageBox.information(self, "��Ϣ����", "������ѧ��!", QMessageBox.Yes) #����
                    else:
                        self.label_infor_tips.setText("")
                        self.Num = self.lineEdit_Num.text()

                        self.Bridge = True
                        print(self.Name,self.Pro,self.Num)

    #�ύ��ť �ۺ���   ������Ϣ
    def save_data(self):
        if self.Bridge == False:
            QMessageBox.information(self, "��Ϣ����", "���ȵ����ʼ���Ժ����ύ������Ϣ!", QMessageBox.Yes) #����
        else:
            self.Sco = int(self.lcdNumber_01.value())
            with open("data.csv", "a", encoding="utf-8-sig", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([self.Name,self.Pro,self.Num,self.Sco])

            self.Bridge = False

            #�ύ�����ݺ���տ������ݷ����´���д
            self.lineEdit_input.setText("00")
            self.lineEdit_Name.setText("")
            self.lineEdit_Proj.setText("")
            self.lineEdit_Num.setText("")

            self.lcdNumber_01.display(00)

            print('�����ѱ���!!!')

    #���ݲ鿴
    def slot_child(self):
        # ��ȡ��������

        # ͨ������Ŀ�͸�������λ����ʾ

        self.one = MyChildWindow()
        self.one.test()
        desktop = QApplication.desktop()
        self.one.move(int(desktop.width() * 0.7), int(desktop.height() * 0.2))
        self.one.show()

    #��������
    def load_csv_data(self, filename,table_name):
        # ��CSV�ļ�
        with open(filename) as csvfile:
            # ��ȡCSV�ļ��е�����
            reader = csv.reader(csvfile)
            data = [row for row in reader]

        # ���ñ����к�����
        table_name.setRowCount(len(data))
        table_name.setColumnCount(len(data[0]))

        # ��������ӵ������
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