# coding=gbk
import csv
from PyQt5 import QtWidgets, QtCore
from target import MyMainWindow

class MainWindow01(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ����QTableWidget���󲢽�������Ϊ���Ĵ��ڲ���
        self.table_widget = QtWidgets.QTableWidget()
        self.setCentralWidget(self.table_widget)

        # �����ڴ�С�����¼������С���Ĳۺ���������
        self.resizeEvent = self.table_resize_event

    def test(self):
        self.shishi = MyMainWindow.load_csv_data(self,'data.csv',self.table_widget)
    """def load_csv_data(self, filename):
        # ��CSV�ļ�
        with open(filename) as csvfile:
            # ��ȡCSV�ļ��е�����
            reader = csv.reader(csvfile)
            data = [row for row in reader]

        # ���ñ����к�����
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data[0]))

        # ��������ӵ������
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                self.table_widget.setItem(i, j, QtWidgets.QTableWidgetItem(col))"""

    def table_resize_event(self, event):
        # ��ȡ���ڴ�С
        width = event.size().width()
        height = event.size().height()

        # ���ñ��Ĵ�С
        self.table_widget.setFixedSize(QtCore.QSize(width, height))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow01()
    window.test()
    # window.load_csv_data('data.csv')
    # window.table_resize_event()
    window.show()
    app.exec_()



