from PySide6.QtWidgets import (QMainWindow,
                               QTabWidget)
from Scripts.VehicleDetectionSystem import VehicleDetectionSystem
from Scripts.UI.ConfigurationWidget import ConfigurationWidget
from Scripts.UI.DetectionWidget import DetectionWidget
from Scripts.UI.VisualizationWidget import VisualizationWidget


class MainApplication(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.vehicleDetectionSystem: VehicleDetectionSystem = VehicleDetectionSystem()
        self.setGeometry(100, 100, 1400, 900)

        self.configurationPage: ConfigurationWidget = ConfigurationWidget(self.vehicleDetectionSystem.detectionPlan)
        self.detectionPage: DetectionWidget = DetectionWidget(self.vehicleDetectionSystem)
        self.visualizationPage: VisualizationWidget = VisualizationWidget(self.vehicleDetectionSystem.result)

        self.detectionPage.SetVisualizationWidget(self.visualizationPage)
        self.mainWidget: QTabWidget = QTabWidget()
        self.mainWidget.setTabsClosable(False)
        self.mainWidget.setMovable(True)

        self.mainWidget.addTab(self.configurationPage, "配置")
        self.mainWidget.addTab(self.detectionPage, "检测")
        self.mainWidget.addTab(self.visualizationPage, "可视化")

        self.setCentralWidget(self.mainWidget)
