# =====
# 1、导入包=====
import sys  #导入sys模块
import ShowMain #导入ShowMain窗体模块(用QtDesigner设计的)
from PyQt5.QtWidgets import * #导入PyQt5的QtWidgets(Qt小部件)相关模块组件
import csv #导入csv模块

#=====2、显示图形用户界面=====
app=QApplication(sys.argv)
MainWindow=QMainWindow() #创建应用程序实例

ui=ShowMain.Ui_MainWindow() #创建主窗体实例

ui.setupUi(MainWindow) #创建已设计窗体ShowMain实例

MainWindow.setFixedSize(MainWindow.width(), MainWindow.height()) #把ShowMain与主窗体进行结合

MainWindow.show() #屏蔽掉主窗体最大化按钮
#显示结合后的主窗体
#=====


# =====#   ===将csv文件内容显示到表格控件QTableView中
# 3.1、打开csv文件并读取到列表中===
csv_file=csv.reader(open('沪深300指数历史交易数据.csv','r')) #以只读方式打开csv文件

csv_list=[] #定义存储整个csv文件的列表

for line in csv_file:
    csv_list.append(line) #按行对csv文件进行循环   #按行将csv文件读取到列表中

# 3.2、设置表格控件的行列数===
RowCount=len(csv_list) #得到csv文件的行数
ColCount=len(csv_list[0]) #得到csv文件的列数

ui.tableWidget.setRowCount(RowCount) #设置表格控件的行数
ui.tableWidget.setColumnCount(ColCount) #设置表格控件的列数#   ===

# 3.3、显示表格控件表头===
Hheader_list=[] #定义存储水平方向表头的列表

for Col in range(0,ColCount,1): #按列进行循环
  Hheader_list.append('%s'%(csv_list[0][Col])) #给水平方向表头列表赋值
ui.tableWidget.setHorizontalHeaderLabels(Hheader_list) #设置水平方向表头标签


Vheader_list=[] #定义存储垂直方向表头的列表

for Row in range(0,RowCount,1): #按行进行循环
  Vheader_list.append('%s%d%s'%('第',Row+1,'行'))   #给垂直方向表头列表赋值
ui.tableWidget.setVerticalHeaderLabels(Vheader_list)   #设置垂直方向的表头标签#

# 3.4、显示表格控件数据===
for Row in range(0,RowCount,1): #按行进行循环
  for Col in range(0,ColCount,1): #按列进行循环
    ui.tableWidget.setItem(Row,Col,QTableWidgetItem(csv_list[Row][Col])) #为每个表格添加数据#=====

  # 4、系统退出命令=====
  sys.exit(app.exec_()) #系统接收退出命令后，退出