

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.input_button = QtWidgets.QPushButton(self.centralwidget)
        self.input_button.setGeometry(QtCore.QRect(270, 320, 93, 28))
        self.input_button.setObjectName("input_button")
        self.label1 = QtWidgets.QLabel(self.centralwidget)
        self.label1.setGeometry(QtCore.QRect(40, 50, 201, 61))
        self.label1.setObjectName("label1")
        self.label2 = QtWidgets.QLabel(self.centralwidget)
        self.label2.setGeometry(QtCore.QRect(40, 150, 421, 81))
        self.label2.setObjectName("label2")
        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(580, 0, 211, 531))
        self.image.setText("")
        self.image.setPixmap(QtGui.QPixmap("line-symmetry-point-geometric-abstraction-pattern-png-favpng-10u5cq377cJpgRqsjAsjz3sxz.jpg"))
        self.image.setObjectName("image")
        self.input_label = QtWidgets.QLabel(self.centralwidget)
        self.input_label.setGeometry(QtCore.QRect(50, 320, 141, 31))
        self.input_label.setObjectName("input_label")
        self.output_label = QtWidgets.QLabel(self.centralwidget)
        self.output_label.setGeometry(QtCore.QRect(50, 380, 141, 31))
        self.output_label.setObjectName("output_label")
        self.output_button = QtWidgets.QPushButton(self.centralwidget)
        self.output_button.setGeometry(QtCore.QRect(270, 380, 93, 28))
        self.output_button.setObjectName("output_button")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.input_button.setText(_translate("MainWindow", "Browse..."))
        self.label1.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:24pt; font-weight:600;\">FaceSpot</span></p></body></html>"))
        self.label2.setText(_translate("MainWindow", "<html><head/><body><p>A detection software dedicated to expose DeepFake-generated content.</p><p>Simply select a video or image file and start the process.</p><p>The accepted files formats are .jpg for images and .mp4 for videos.</p></body></html>"))
        self.input_label.setText(_translate("MainWindow", "No input file selected"))
        self.output_label.setText(_translate("MainWindow", "No output path selected"))
        self.output_button.setText(_translate("MainWindow", "Browse..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
