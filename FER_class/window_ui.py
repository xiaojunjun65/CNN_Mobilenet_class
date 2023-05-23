import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QPushButton, QTextEdit, QMessageBox, QFileDialog
from PyQt5.QtGui import QPixmap, QColor, QPalette,QFont
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np
import cv2
class EmotionRecognitionSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("表情识别系统")
        self.setGeometry(100, 100, 720, 350)
        self.class_names =['anger', 'contempt', 'disgust','fear', 'happy', 'sadness','surprise']
        # 设置整体背景颜色为浅蓝色
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(200, 220, 255))
        self.setPalette(palette)
        self.img_path =None
        # 创建下拉框
        self.network_dropdown = QComboBox(self)
        self.network_dropdown.addItem("CNN")
        self.network_dropdown.addItem("MobileNetV2")
        self.network_dropdown.setGeometry(20, 20, 200, 30)

        # 创建Label显示选择的图片
        self.selected_image_label = QLabel("选择的图片", self)
        self.selected_image_label.setGeometry(50, 70, 144, 144)

        # 创建按钮：选择图片
        self.select_image_button = QPushButton("选择图片", self)
        self.select_image_button.setGeometry(20, 250, 200, 30)
        self.select_image_button.clicked.connect(self.select_image)


        self.exit = QPushButton("退出", self)
        self.exit.setGeometry(580, 310, 60, 30)
        self.exit.clicked.connect(self.closeEvent)

        # 创建按钮：开始推理
        self.start_inference_button = QPushButton("开始推理", self)
        self.start_inference_button.setGeometry(20, 290, 200, 30)
        self.start_inference_button.clicked.connect(self.start_inference)
        # 创建日志框
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setGeometry(300, 10, 400, 300)
        font = QFont("Arial", 12)  # 设置字体为Arial，大小为16
        self.log_text_edit.setFont(font)

    def select_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Image files (*.jpg *.jpeg *.png)")
        self.img_path =file_path
        if file_path:
            # 显示选择的图片
            pixmap = QPixmap(file_path).scaled(144, 144, Qt.KeepAspectRatio)
            self.selected_image_label.setPixmap(pixmap)
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
            exit()
        else:
            event.ignore()

    def start_inference(self):
        selected_network = self.network_dropdown.currentText()
        # 在这里添加开始推理的代码，根据选择的网络模型进行相应的操作
        if selected_network =="CNN":
            model = tf.keras.models.load_model("../models/cnn.h5")
        else:
            model = tf.keras.models.load_model("../models/mobilenet.h5")
        img = cv2.imread(self.img_path)
        img = cv2.resize(img, (48, 48))
        img = img.reshape(1, 48, 48, 3)
        outputs = model.predict(img)
        acc = outputs
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        jieguo = """
        ==========================
        
        当前网络模型：{}，
        
        
        当前图片情绪：{}
        
        
        ==========================
        
        """.format(selected_network,result)
        self.log_text_edit.clear()
        self.log_text_edit.append(jieguo)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionRecognitionSystem()
    window.show()
    sys.exit(app.exec_())
