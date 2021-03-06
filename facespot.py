import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QIcon, QColor, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QAction, QMessageBox
from PyQt5.QtWidgets import QCalendarWidget, QFontDialog, QColorDialog, QTextEdit, QFileDialog
from PyQt5.QtWidgets import QCheckBox, QProgressBar, QComboBox, QLabel, QStyleFactory, QLineEdit, QInputDialog
from predict import predict

'''
Main program that presents the prediction code in the form of a graphical user interface.
'''

# pyuic5 -x testing.ui -o testing3.py
class myWindow(QMainWindow):

    def __init__(self):
        super(myWindow, self).__init__()
        self.setGeometry(300, 200, 800, 600) # starting x, starting y, width, height
        self.setWindowTitle("FaceSpot")
        self.UI()

    def UI(self):

        '''
        Placing UI elements.
        '''

        self.label1 = QtWidgets.QLabel(self)
        self.label1.setGeometry(QtCore.QRect(40, 50, 201, 61))
        self.label1.setObjectName("label1")

        self.label2 = QtWidgets.QLabel(self)
        self.label2.setGeometry(QtCore.QRect(40, 150, 421, 81))
        self.label2.setObjectName("label2")

        self.input_label = QtWidgets.QLabel(self)
        self.input_label.setGeometry(QtCore.QRect(50, 320, 240, 31))
        self.input_label.setObjectName("input_label")

        self.input_button = QtWidgets.QPushButton(self)
        self.input_button.setGeometry(QtCore.QRect(450, 320, 93, 28))
        self.input_button.setObjectName("input_button")
        self.filepath = self.input_button.clicked.connect(self.openFile)

        self.file_label = QtWidgets.QLabel(self)
        self.file_label.setGeometry(QtCore.QRect(50, 380, 240, 31))
        self.file_label.setObjectName("file_label")

        self.test_button = QtWidgets.QPushButton(self)
        self.test_button.setGeometry(QtCore.QRect(450, 380, 93, 28))
        self.test_button.setObjectName("test_button")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self.startPredict)

        self.image = QtWidgets.QLabel(self)
        self.image.setGeometry(QtCore.QRect(590, 10, 211, 561))
        self.image.setPixmap(QtGui.QPixmap("ui/bg.png"))
        self.image.setScaledContents(True)
        self.image.setObjectName("image")

        self.predict_label = QtWidgets.QLabel(self)
        self.predict_label.setGeometry(QtCore.QRect(50, 440, 240, 31))
        self.predict_label.setObjectName("predict_label")

        self.loading_label = QtWidgets.QLabel(self)
        self.loading_label.setGeometry(QtCore.QRect(50, 500, 240, 31))
        self.loading_label.setObjectName("loading_label")


        _translate = QtCore.QCoreApplication.translate
        self.label1.setText(_translate("MainWindow",
                                       "<html><head/><body><p><span style=\" font-size:24pt; font-weight:600;\">FaceSpot</span></p></body></html>"))
        self.label2.setText(_translate("MainWindow",
                                       "<html><head/><body><p>A detection software dedicated to expose DeepFake-generated content.</p><p>Simply "
                                       "select a video or image file and start the process.</p><p>The accepted files formats are .jpg/.png for images and "
                                       ".mp4 for videos.</p></body></html>"))
        self.input_label.setText(_translate("MainWindow", "No input file selected"))
        self.input_button.setText(_translate("MainWindow", "Browse..."))
        self.test_button.setText(_translate("MainWindow", "Test"))
        self.loading_label.setText(_translate("MainWindow", "Program not running."))


    def openFile(self):

        '''
        Opens a file and updates relevant UI labels.
        '''

        _translate = QtCore.QCoreApplication.translate
        path, _ = QFileDialog.getOpenFileName(self, "Open File")
        if path.endswith(".mp4") or path.endswith(".jpg") or path.endswith(".png"):
            self.input_label.setText(_translate("MainWindow", "A valid file has been selected"))
            self.file_label.setText(_translate("MainWindow", path))
            self.file_label.adjustSize()
            self.test_button.setEnabled(True)
            self.filepath = path
        else:
            self.input_label.setText(_translate("MainWindow", "Please select a valid file format (jpg/png/mp4)"))
            self.file_label.setText(_translate("MainWindow", path))
            self.file_label.adjustSize()
            self.test_button.setEnabled(False)
            return None


    def startPredict(self):

        '''
        Starts the prediction process updates relevant UI labels.
        '''

        print(self.filepath)
        _translate = QtCore.QCoreApplication.translate
        self.loading_label.setText(_translate("MainWindow", "Please wait.. The program is currently running."))
        self.loading_label.adjustSize()
        self.loading_label.repaint()
        deepfaked = predict(self.filepath)
        if deepfaked:
            self.predict_label.setText("Prediction successful." + self.filepath + " has been manipulated.")
            self.predict_label.adjustSize()
            self.loading_label.setText("Done.")
            self.loading_label.adjustSize()
        else:
            self.predict_label.setText("Prediction successful." + self.filepath + " is unmodified.")
            self.predict_label.adjustSize()
            self.loading_label.setText("Done.")
            self.loading_label.adjustSize()


def window():
    app = QApplication(sys.argv) # config window based on OS
    win = myWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    window()