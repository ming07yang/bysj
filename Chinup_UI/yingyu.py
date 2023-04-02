# coding=gbk
import csv
from PyQt5 import QtWidgets, QtCore
from target import MyMainWindow

class MainWindow01(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 创建QTableWidget对象并将其设置为中心窗口部件
        self.table_widget = QtWidgets.QTableWidget()
        self.setCentralWidget(self.table_widget)

        # 将窗口大小更改事件与表格大小更改槽函数相连接
        self.resizeEvent = self.table_resize_event

    def test(self):
        self.shishi = MyMainWindow.load_csv_data(self,'data.csv',self.table_widget)
    """def load_csv_data(self, filename):
        # 打开CSV文件
        with open(filename) as csvfile:
            # 读取CSV文件中的数据
            reader = csv.reader(csvfile)
            data = [row for row in reader]

        # 设置表格的行和列数
        self.table_widget.setRowCount(len(data))
        self.table_widget.setColumnCount(len(data[0]))

        # 将数据添加到表格中
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                self.table_widget.setItem(i, j, QtWidgets.QTableWidgetItem(col))"""

    def table_resize_event(self, event):
        # 获取窗口大小
        width = event.size().width()
        height = event.size().height()

        # 设置表格的大小
        self.table_widget.setFixedSize(QtCore.QSize(width, height))

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow01()
    window.test()
    # window.load_csv_data('data.csv')
    # window.table_resize_event()
    window.show()
    app.exec_()



