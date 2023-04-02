# coding=gbk
import sys
import csv
import cv2
import pandas as pd
from PyQt5.QtWidgets import QMessageBox, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
from test011 import Ui_MainWindow
from Child import Ui_ChildWindow
from target import MyMainWindow
class MyChildWindow(QMainWindow,Ui_ChildWindow):  #
    def __init__(self, parent=None):
        super(MyChildWindow, self).__init__(parent)
        self.setupUi(self)

    def test(self):
        self.shiyixia = MyMainWindow.load_csv_data(self,'data.csv',self.Data_Table)
        pass


if __name__=="__main__":
    app = QApplication(sys.argv)
    mainWindow = MyChildWindow()
    # test = Ui_ChildWindow()
    mainWindow.test()
    # ui = test011.Ui_MainWindow()
    # ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())