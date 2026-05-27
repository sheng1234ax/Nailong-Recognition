import sys
import os
import cv2
import torch
import numpy as np
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5 import uic


class DetectionThread(QThread):
    frame_ready = pyqtSignal(QPixmap, QPixmap)
    log_msg = pyqtSignal(str)

    def __init__(self, model, source_path, target_class='nailong', is_video=True):
        super().__init__()
        self.model = model
        self.source_path = source_path
        self.target_class = target_class
        self.is_video = is_video
        self.running = True

    def run(self):
        if self.is_video:
            cap = cv2.VideoCapture(self.source_path)
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame)
                det_df = results.pandas().xyxy[0]

                target_count = len(det_df[det_df['name'] == self.target_class])
                if target_count > 0:
                    self.log_msg.emit(f"画面中检测到 {target_count} 个 [{self.target_class}]")

                res_frame = np.squeeze(results.render())
                raw_pixmap = self.convert_cv_qt(frame)
                res_pixmap = self.convert_cv_qt(res_frame)

                self.frame_ready.emit(raw_pixmap, res_pixmap)

                self.msleep(30)

            cap.release()
            self.log_msg.emit("视频处理完毕。")

    def stop(self):
        self.running = False
        self.wait()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)


# ==========================================
# 核心组件二：主窗口控制器
# ==========================================
class MainController:
    def __init__(self):
        if not os.path.exists("nailong.ui"):
            print("致命错误: 找不到 nailong.ui 文件！")
            sys.exit(-1)

        self.ui = uic.loadUi("nailong.ui")

        # 初始化变量
        self.current_source = None
        self.is_video = False
        self.thread = None
        self.target_class = 'nailong'

        self.init_yolo_model()

        # 绑定按钮
        self.ui.photoButton.clicked.connect(self.select_photo)
        self.ui.videoButton.clicked.connect(self.select_video)
        self.ui.startButton.clicked.connect(self.start_recognition)
        self.ui.stopButton.clicked.connect(self.stop_recognition)

    def init_yolo_model(self):
        self.append_log("正在配置 YOLOv5 推理核心...")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        yolov5_source_dir = os.path.join(current_dir, 'yolov5')
        weight_path = os.path.join(current_dir, 'nailong.pt')

        if os.path.exists(yolov5_source_dir) and os.path.exists(weight_path):
            try:
                self.append_log("状态：检测到完整的本地框架与权重，正在从本地加载自定义模型...")
                self.model = torch.hub.load(yolov5_source_dir, 'custom', path=weight_path, source='local', force_reload=True)

                self.model.conf = 0.5
                self.target_class = 'nailong'
                self.append_log("成功：自定义本地模型部署就绪！\n")
            except Exception as e:
                self.append_log(f"警告：本地加载异常 ({str(e)})。正在强制切换为在线官方预训练模型...")
                self.load_official_model()
        else:
            self.append_log("状态：未找到本地模型文件或检测源码目录。正在联网获取官方预训练权重...")
            self.load_official_model()

    def load_official_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            self.target_class = 'people'
            self.append_log("成功：已成功挂载官方 yolov5s 预训练模型（测试靶标自动调整为: people）！\n")
        except Exception as e:
            self.append_log(f"致命错误：联网下载官方模型失败。错误信息: {str(e)}")
            self.model = None

    def select_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(self.ui, "选择待测图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.current_source = file_path
            self.is_video = False
            self.append_log(f"已选择图片: {file_path}")
            pixmap = QPixmap(file_path)
            self.ui.inputput_label.setPixmap(pixmap.scaled(self.ui.inputput_label.size(), Qt.KeepAspectRatio))

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self.ui, "选择待测视频", "", "Videos (*.mp4 *.avi *.mov)")
        if file_path:
            self.current_source = file_path
            self.is_video = True
            self.append_log(f"已选择视频: {file_path}")

    def start_recognition(self):
        if not getattr(self, 'model', None):
            self.append_log("错误：检测核心未成功建立，无法启动识别。")
            return

        if not self.current_source:
            self.append_log("提示：请先选择有效的输入源（图片/视频）。")
            return

        if self.is_video:
            self.thread = DetectionThread(self.model, self.current_source, target_class=self.target_class,
                                          is_video=True)
            self.thread.frame_ready.connect(self.update_display)
            self.thread.log_msg.connect(self.append_log)
            self.thread.start()
            self.append_log(f"异步线程已启动，正在扫描目标 [{self.target_class}]...")
        else:
            img = cv2.imread(self.current_source)
            results = self.model(img)

            det_df = results.pandas().xyxy[0]
            target_count = len(det_df[det_df['name'] == self.target_class])
            self.append_log(f"静态推理完毕。共计发现 {target_count} 个 [{self.target_class}]")

            res_img = np.squeeze(results.render())
            dummy_thread = DetectionThread(self.model, None)
            res_pixmap = dummy_thread.convert_cv_qt(res_img)
            self.ui.output_label.setPixmap(res_pixmap.scaled(self.ui.output_label.size(), Qt.KeepAspectRatio))

    def stop_recognition(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.append_log("用户中断：视频流识别已挂起。")
        else:
            self.append_log("提示：当前没有活跃的后台识别任务。")

    def update_display(self, raw_pixmap, res_pixmap):
        self.ui.inputput_label.setPixmap(raw_pixmap.scaled(self.ui.inputput_label.size(), Qt.KeepAspectRatio))
        self.ui.output_label.setPixmap(res_pixmap.scaled(self.ui.output_label.size(), Qt.KeepAspectRatio))

    def append_log(self, text):
        current_text = self.ui.infomation_output.text()
        if len(current_text) > 3000:
            current_text = current_text[-1500:]
        self.ui.infomation_output.setText(current_text + text + "\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = MainController()
    controller.ui.show()
    sys.exit(app.exec_())