# coding=gbk
import sys
import csv
from concurrent.futures import ThreadPoolExecutor

import cv2
import pandas as pd
from PyQt5.QtCore import QRunnable, QThreadPool
from PyQt5.QtWidgets import QMessageBox, QDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow
from ui_wind.test011 import Ui_MainWindow
from ui_child.Child import Ui_ChildWindow
# from detect.chinup_det_keypoint_unite_infer import device

import argparse
import os
import json
import cv2
import math
import numpy as np
import paddle
import yaml

import time
from detect.common import Triangle, Point, Get_angle

from detect.det_keypoint_unite_utils import argsparser
from detect.preprocess import decode_image
from detect.infer import Detector, DetectorPicoDet, PredictConfig, print_arguments, get_test_images, bench_log
from detect.keypoint_infer import KeyPointDetector, PredictConfig_KeyPoint
from detect.visualize import visualize_pose
from detect.benchmark_utils import PaddleInferBenchmark
from detect.utils import get_current_memory_mb
from detect.keypoint_postprocess import translate_to_ori_images


class Worker(QRunnable):  # 多线程
    def __init__(self,threadFlag,func):
    # def __int__(self,*args, **kwargs):
        super().__init__()
        self.threadFlag = threadFlag
        self.func = func

    def run(self):
        if self.threadFlag:
            self.func()
    def close(self):
        self.threadFlag = False

class MyMainWindow(QMainWindow,Ui_MainWindow):  #
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.Flag = False
        
#-------------------------------------------------------------------------------------------
    KEYPOINT_SUPPORT_MODELS = {
        'HigherHRNet': 'keypoint_bottomup',
        'HRNet': 'keypoint_topdown'
    }

    def predict_with_given_det(self,image, det_res, keypoint_detector,
                               keypoint_batch_size, run_benchmark):
        rec_images, records, det_rects = keypoint_detector.get_person_from_rect(
            image, det_res)
        keypoint_vector = []
        score_vector = []

        rect_vector = det_rects
        keypoint_results = keypoint_detector.predict_image(
            rec_images, run_benchmark, repeats=10, visual=False)
        keypoint_vector, score_vector = translate_to_ori_images(keypoint_results,
                                                                np.array(records))
        keypoint_res = {}
        keypoint_res['keypoint'] = [
            keypoint_vector.tolist(), score_vector.tolist()
        ] if len(keypoint_vector) > 0 else [[], []]
        keypoint_res['bbox'] = rect_vector
        return keypoint_res

    def topdown_unite_predict(self,detector,
                              topdown_keypoint_detector,
                              image_list,
                              keypoint_batch_size=1,
                              save_res=False):
        det_timer = detector.get_timer()
        store_res = []
        for i, img_file in enumerate(image_list):
            # Decode image in advance in det + pose prediction
            det_timer.preprocess_time_s.start()
            image, _ = decode_image(img_file, {})
            det_timer.preprocess_time_s.end()

            if FLAGS.run_benchmark:
                results = detector.predict_image(
                    [image], run_benchmark=True, repeats=10)

                cm, gm, gu = get_current_memory_mb()
                detector.cpu_mem += cm
                detector.gpu_mem += gm
                detector.gpu_util += gu
            else:
                results = detector.predict_image([image], visual=False)
            results = detector.filter_box(results, FLAGS.det_threshold)
            if results['boxes_num'] > 0:
                keypoint_res = self.predict_with_given_det(
                    image, results, topdown_keypoint_detector, keypoint_batch_size,
                    FLAGS.run_benchmark)

                if save_res:
                    save_name = img_file if isinstance(img_file, str) else i
                    store_res.append([
                        save_name, keypoint_res['bbox'],
                        [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                    ])
            else:
                results["keypoint"] = [[], []]
                keypoint_res = results
            if FLAGS.run_benchmark:
                cm, gm, gu = get_current_memory_mb()
                topdown_keypoint_detector.cpu_mem += cm
                topdown_keypoint_detector.gpu_mem += gm
                topdown_keypoint_detector.gpu_util += gu
            else:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                visualize_pose(
                    img_file,
                    keypoint_res,
                    visual_thresh=FLAGS.keypoint_threshold,
                    save_dir=FLAGS.output_dir)
        if save_res:
            """
            1) store_res: a list of image_data
            2) image_data: [imageid, rects, [keypoints, scores]]
            3) rects: list of rect [xmin, ymin, xmax, ymax]
            4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
            5) scores: mean of all joint conf
            """
            with open("det_keypoint_unite_image_results.json", 'w') as wf:
                json.dump(store_res, wf, indent=4)

    global chinUpCnt

    def topdown_unite_predict_video(self,detector,
                                    topdown_keypoint_detector,
                                    camera_id,
                                    keypoint_batch_size=1,
                                    save_res=False):
        video_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(FLAGS.video_file)
            # capture = cv2.VideoCapture('/home/shine/Desktop/mmpose-master/demo/chinup.mp4')
            video_name = os.path.split(FLAGS.video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        out_path = os.path.join(FLAGS.output_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        # writer = cv2.VideoWriter("/home/shine/PaddleDetection/output/out.mp4", fourcc, fps, (width, height))
        index = 0
        store_res = []

        direction = 0  # 判断方向
        chinUpCnt = 0
        timeC = 0
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            index += 1
            # print('detect frame: %d' % (index))

            # 进行跳帧处理
            timeC += 1
            if (timeC % 3 != 0):
                frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                continue

            results = detector.predict_image([frame2], visual=False)
            results = detector.filter_box(results, FLAGS.det_threshold)
            if results['boxes_num'] == 0:
                writer.write(frame)
                continue

            keypoint_res = self.predict_with_given_det(
                frame2, results, topdown_keypoint_detector, keypoint_batch_size,
                FLAGS.run_benchmark)

            pose = keypoint_res['keypoint']
            pose = np.array(pose[0])

            # print(pose)
            HEAD_x = pose[0, 0, 0]
            HEAD_y = pose[0, 0, 1]

            LEFT_ANKLE_x = pose[0, 15, 0]
            LEFT_ANKLE_y = pose[0, 15, 1]
            LEFT_KNEE_x = pose[0, 13, 0]
            LEFT_KNEE_y = pose[0, 13, 1]
            LEFT_HIP_x = pose[0, 11, 0]
            LEFT_HIP_y = pose[0, 11, 1]
            RIGHT_ANKLE_x = pose[0, 16, 0]
            RIGHT_ANKLE_y = pose[0, 16, 1]
            RIGHT_KNEE_x = pose[0, 14, 0]
            RIGHT_KNEE_y = pose[0, 14, 1]
            RIGHT_HIP_x = pose[0, 12, 0]
            RIGHT_HIP_y = pose[0, 12, 1]

            LEFT_WRIST_x = pose[0, 9, 0]
            LEFT_WRIST_y = pose[0, 9, 1]
            LEFT_ELBOW_x = pose[0, 7, 0]
            LEFT_ELBOW_y = pose[0, 7, 1]
            LEFT_SHOULDER_x = pose[0, 5, 0]
            LEFT_SHOULDER_y = pose[0, 5, 1]
            RIGHT_WRIST_x = pose[0, 10, 0]
            RIGHT_WRIST_y = pose[0, 10, 1]
            RIGHT_ELBOW_x = pose[0, 8, 0]
            RIGHT_ELBOW_y = pose[0, 8, 1]
            RIGHT_SHOULDER_x = pose[0, 6, 0]
            RIGHT_SHOULDER_y = pose[0, 6, 1]

            LEFT_ELBOW_t = Triangle(Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y),
                                    Point(LEFT_ELBOW_x, LEFT_ELBOW_y),
                                    Point(LEFT_WRIST_x, LEFT_WRIST_y))
            RIGHT_ELBOW_t = Triangle(Point(RIGHT_SHOULDER_x, RIGHT_SHOULDER_y),
                                     Point(RIGHT_ELBOW_x, RIGHT_ELBOW_y),
                                     Point(RIGHT_WRIST_x, RIGHT_WRIST_y))
            SHOULDER_t = Triangle(Point(LEFT_ELBOW_x, LEFT_ELBOW_y),
                                  Point(LEFT_SHOULDER_x, LEFT_SHOULDER_y),
                                  Point(LEFT_HIP_x, LEFT_HIP_y))

            xArray = np.array([LEFT_SHOULDER_x, LEFT_HIP_x, LEFT_KNEE_x])
            # 求肩膀，腰和膝盖的标准偏差，越小越偏向于直立
            xStdInt = np.std(xArray)

            threshold_x = 70

            if (float(xStdInt) < threshold_x) & (LEFT_WRIST_y < LEFT_ELBOW_y):

                # 两臂弯曲程度
                elbow_angle = LEFT_ELBOW_t.angle_p2()

                # 两个先验角度35~170
                if elbow_angle >= 170:
                    if direction == 0:
                        chinUpCnt = chinUpCnt + 0.5
                        direction = 1
                if elbow_angle <= 35:
                    if direction == 1:
                        if HEAD_y < LEFT_WRIST_y:
                            chinUpCnt = chinUpCnt + 0.5
                            direction = 0

                            self.lcdNumber_01.display(chinUpCnt)  #连接  wind - lcdNumber
                            print(self.lcdNumber_01.value())
                            # print("chinUpCnt:" + str(int(chinUpCnt)))

                '''
                # 肩膀弯曲角度
                shoulder = SHOULDER_t.angle_p2()
                print("shoulder_before", shoulder_before, "shoulder", shoulder, "frameCnt_ChinUp", frameCnt_ChinUp)
                if shoulder is not None:
                    if ((shoulder > 90) & (shoulder_before < 90)) | ((shoulder < 90) & (shoulder_before > 90)):
                        frameCnt_ChinUp += 1
                        # 防止抖动影响
                        if frameCnt_ChinUp < threshold_wave:
                            chinUpCnt = chinUpCnt
                        else:
                            chinUpCnt += 1
                            frameCnt_ChinUp = 0
                    # 反转上一帧角度
                    if shoulder_before == 180:
                        shoulder_before = 0
                    else:
                        shoulder_before = 180
                    print('CHINCOUNT:', chinUpCnt)
                else:
                    frameCnt_ChinUp = 0
            else:
                frameCnt_ChinUp = 0
                '''

            im = visualize_pose(
                frame,
                keypoint_res,
                visual_thresh=FLAGS.keypoint_threshold,
                returnimg=True)
            if save_res:
                store_res.append([
                    index, keypoint_res['bbox'],
                    [keypoint_res['keypoint'][0], keypoint_res['keypoint'][1]]
                ])

            # writer.write(im)
            if camera_id != -1:
                # 将帧转换成Qt图像格式
                image = QtGui.QImage(im.data, im.shape[1], im.shape[0],
                                     QtGui.QImage.Format_RGB888).rgbSwapped()
                # 在标签上显示图片
                self.label_vedio.setPixmap(QtGui.QPixmap.fromImage(image))

                # cv2.imshow('Mask Detection', im)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

            # present the video and count
            frame = cv2.resize(im, (720, 480))
            cv2.putText(frame, "chinupcount:" + str(int(chinUpCnt)) + " frame:" + str(index), (20, 30),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            # self.show_videob(ret,frame)
            # 将帧转换成Qt图像格式
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            # 在标签上显示图片
            self.label_vedio.setPixmap(QtGui.QPixmap.fromImage(image))

            # cv2.namedWindow('chinup')
            # cv2.imshow("chinup", frame)
            # keycode = cv2.waitKey(1)
            # if keycode == 27:
            #     cv2.destroyWindow('chinup')
            #     cv2.videoCapture.release()
            #     break
            writer.write(frame)
        writer.release()
        print('output_video saved to: {}'.format(out_path))
        if save_res:
            """
            1) store_res: a list of frame_data
            2) frame_data: [frameid, rects, [keypoints, scores]]
            3) rects: list of rect [xmin, ymin, xmax, ymax]
            4) keypoints: 17(joint numbers)*[x, y, conf], total 51 data in list
            5) scores: mean of all joint conf
            """
            with open("det_keypoint_unite_video_results.json", 'w') as wf:
                json.dump(store_res, wf, indent=4)

    def main(self):
        deploy_file = os.path.join(FLAGS.det_model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        arch = yml_conf['arch']
        detector_func = 'Detector'
        if arch == 'PicoDet':
            detector_func = 'DetectorPicoDet'

        detector = eval(detector_func)(FLAGS.det_model_dir,
                                       device=FLAGS.device,
                                       run_mode=FLAGS.run_mode,
                                       trt_min_shape=FLAGS.trt_min_shape,
                                       trt_max_shape=FLAGS.trt_max_shape,
                                       trt_opt_shape=FLAGS.trt_opt_shape,
                                       trt_calib_mode=FLAGS.trt_calib_mode,
                                       cpu_threads=FLAGS.cpu_threads,
                                       enable_mkldnn=FLAGS.enable_mkldnn,
                                       threshold=FLAGS.det_threshold)

        topdown_keypoint_detector = KeyPointDetector(
            FLAGS.keypoint_model_dir,
            device=FLAGS.device,
            run_mode=FLAGS.run_mode,
            batch_size=FLAGS.keypoint_batch_size,
            trt_min_shape=FLAGS.trt_min_shape,
            trt_max_shape=FLAGS.trt_max_shape,
            trt_opt_shape=FLAGS.trt_opt_shape,
            trt_calib_mode=FLAGS.trt_calib_mode,
            cpu_threads=FLAGS.cpu_threads,
            enable_mkldnn=FLAGS.enable_mkldnn,
            use_dark=FLAGS.use_dark)
        keypoint_arch = topdown_keypoint_detector.pred_config.arch
        assert self.KEYPOINT_SUPPORT_MODELS[
                   keypoint_arch] == 'keypoint_topdown', 'Detection-Keypoint unite inference only supports topdown models.'

        # predict from video file or camera video stream
        if FLAGS.video_file is not None or FLAGS.camera_id != -1:
            self.topdown_unite_predict_video(detector, topdown_keypoint_detector,
                                        FLAGS.camera_id, FLAGS.keypoint_batch_size,
                                        FLAGS.save_res)
        else:
            # predict from image
            img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
            self.topdown_unite_predict(detector, topdown_keypoint_detector, img_list,
                                  FLAGS.keypoint_batch_size, FLAGS.save_res)
            if not FLAGS.run_benchmark:
                detector.det_times.info(average=True)
                topdown_keypoint_detector.det_times.info(average=True)
            else:
                mode = FLAGS.run_mode
                det_model_dir = FLAGS.det_model_dir
                det_model_info = {
                    'model_name': det_model_dir.strip('/').split('/')[-1],
                    'precision': mode.split('_')[-1]
                }
                bench_log(detector, img_list, det_model_info, name='Det')
                keypoint_model_dir = FLAGS.keypoint_model_dir
                keypoint_model_info = {
                    'model_name': keypoint_model_dir.strip('/').split('/')[-1],
                    'precision': mode.split('_')[-1]
                }
                bench_log(topdown_keypoint_detector, img_list, keypoint_model_info,
                          FLAGS.keypoint_batch_size, 'KeyPoint')
# ---------------------------------------------------------------------------------------------------------------
    def show_video(self,filename):
        # self.cap = cv2.VideoCapture(filename)
        self.cap = cv2.VideoCapture(0)

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
        self.pushButton_start_style.setEnabled(False) #启动按钮 上锁
        self.statusBar().showMessage('已开启系统！',)

        # 创建一个Worker对象，传入要执行的函数
        self.worker = self.Worker(self.Flag,self.main)
        # # 将Worker对象添加到线程池中
        QThreadPool.globalInstance().start(self.worker)
        # self.main()
        # self.show_video('video.mp4')

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
        self.pushButton_start_style.setEnabled(True) #启动按钮解锁
        # self.worker.close()
        # self.worker = Worker(self.Flag0,self.main)
        self.statusBar().showMessage('系统已关闭！', )
        # self.cap.release()
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
            with open("data.csv", "a", encoding="gbk", newline="") as csvfile:
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

#===============================================================
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    t1 = time.time()
    # mainWindow.main()
    t2 = time.time()
    print(t2 - t1, 's')
# =========================================================
    # ui = test011.Ui_MainWindow()
    # ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())

