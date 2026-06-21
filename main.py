import sys
from PySide6.QtWidgets import QApplication
from Scripts.MainApplication import MainApplication


def main():
    """ 程序入口 """
    app: QApplication = QApplication(sys.argv)
    mainWindow: MainApplication = MainApplication()
    mainWindow.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
