import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import (QAbstractTableModel,
                            QModelIndex,
                            QMargins)
from PySide6.QtGui import (QImage,
                           QPixmap,
                           QPainter,
                           Qt, QColor)
from PySide6.QtCharts import (QBarSet,
                              QBarSeries,
                              QChart,
                              QBarCategoryAxis,
                              QValueAxis,
                              QChartView,
                              QPieSeries,
                              QPieSlice, QLegend)
from PySide6.QtWidgets import (QWidget,
                               QVBoxLayout,
                               QHBoxLayout,
                               QSlider,
                               QLabel,
                               QPushButton,
                               QGroupBox,
                               QTabWidget,
                               QMessageBox,
                               QFileDialog,
                               QTableView,
                               QHeaderView)
from Scripts.Lane import Lane
from Scripts.Result import Result


class PandasModel(QAbstractTableModel):
    """Pandas数据模型，用于在QTableView中显示DataFrame数据"""

    def __init__(self, dataFrame: pd.DataFrame) -> None:
        super().__init__()
        self._dataFrame: pd.DataFrame = dataFrame

    def rowCount(self, parent=None) -> int:
        """返回行数"""
        return self._dataFrame.shape[0]

    def columnCount(self, parent=None) -> int:
        """返回列数"""
        return self._dataFrame.shape[1]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> any:
        """返回指定索引的数据"""
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._dataFrame.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> any:
        """返回表头数据"""
        if role != Qt.ItemDataRole.DisplayRole:
            return None

        if orientation == Qt.Orientation.Horizontal:
            return self._dataFrame.columns[section]
        else:
            return str(section + 1)


class VisualizationWidget(QWidget):
    """可视化标签页 - 显示图表和轨迹数据"""

    detectionLabel: QLabel
    statisticsLabel: QLabel
    frameSlider: QSlider
    frameLabel: QLabel

    def __init__(self, result: Result) -> None:
        super().__init__()
        self.result: Result = result
        self.currentFrameIndex: int = 0
        self.mainLayout: QTabWidget = QTabWidget()
        outerLayout: QVBoxLayout = QVBoxLayout(self)
        outerLayout.setContentsMargins(0, 0, 0, 0)
        outerLayout.addWidget(self.mainLayout)

    def ResultUpdate(self, result: Result) -> None:
        """结果更新 - 先清空所有 Tab，再根据 result 动态创建"""
        self.result = result

        while self.mainLayout.count() > 0:
            self.mainLayout.removeTab(0)
        self.CreateControlTab()
        self.CreateTrajectoryTab()  # 内部已自动设置 slider 和 label
        self.CreateDataTab()

        self.currentFrameIndex = 0
        self.UpdateImages(0)

    def CreateControlTab(self) -> None:
        """创建控制台Tab内容"""

        def GetPieChartView(chartTitle: str, nameList: list[str], valueList: list[float]) -> QChartView:
            """创建饼图视图"""
            pieSeriesObject: QPieSeries = QPieSeries()

            for index in range(len(nameList)):
                pieSliceObject: QPieSlice = pieSeriesObject.append(nameList[index], valueList[index])
                pieSliceObject.setLabelVisible(True)
                pieSliceObject.setLabelPosition(QPieSlice.LabelPosition.LabelInsideHorizontal)
                pieSliceObject.setLabel(f"{nameList[index]}: {int(valueList[index])}")
                pieSliceObject.setLabelColor(QColor(255, 255, 255, 255))

            def onHovered(pieSliceItem: QPieSlice, hovered: bool) -> None:
                pieSliceItem.setExploded(hovered)

            pieSeriesObject.hovered.connect(onHovered)
            chartObject: QChart = QChart()
            chartObject.addSeries(pieSeriesObject)
            chartObject.setTitle(chartTitle)
            chartObject.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

            legendObject: QLegend = chartObject.legend()
            legendObject.setAlignment(Qt.AlignmentFlag.AlignBottom)

            chartViewObject: QChartView = QChartView(chartObject)
            chartViewObject.setRenderHint(QPainter.RenderHint.Antialiasing)
            return chartViewObject

        controlWidget: QWidget = QWidget()
        controlLayout: QVBoxLayout = QVBoxLayout(controlWidget)

        distributionLayout: QHBoxLayout = QHBoxLayout()

        vehicleTypeDistributionDataFrame: pd.DataFrame = self.result.GetVehicleTypeDistributionDataFrame()
        vehicleTypeNameList: list[str] = vehicleTypeDistributionDataFrame["车辆类型"].tolist()
        vehicleTypeValueList: list[float] = vehicleTypeDistributionDataFrame["车辆数目"].tolist()
        vehicleTypePieChartView: QChartView = GetPieChartView("车辆类型分布", vehicleTypeNameList, vehicleTypeValueList)
        distributionLayout.addWidget(vehicleTypePieChartView)

        vehicleColorDistributionDataFrame: pd.DataFrame = self.result.GetVehicleColorDistributionDataFrame()
        vehicleColorNameList: list[str] = vehicleColorDistributionDataFrame["车辆颜色"].tolist()
        vehicleColorValueList: list[float] = vehicleColorDistributionDataFrame["车辆数目"].tolist()
        vehicleColorPieChartView: QChartView = GetPieChartView("车辆颜色分布", vehicleColorNameList, vehicleColorValueList)
        distributionLayout.addWidget(vehicleColorPieChartView)

        vehicleLaneDistributionDataFrame: pd.DataFrame = self.result.GetVehicleLaneDistributionDataFrame()
        vehicleLaneNameList: list[str] = vehicleLaneDistributionDataFrame["车辆车道"].tolist()
        vehicleLaneValueList: list[float] = vehicleLaneDistributionDataFrame["车辆数目"].tolist()
        vehicleLanePieChartView: QChartView = GetPieChartView("车辆所在车道分布", vehicleLaneNameList, vehicleLaneValueList)
        distributionLayout.addWidget(vehicleLanePieChartView)

        controlLayout.addLayout(distributionLayout)

        exportAllButtonLayout: QHBoxLayout = QHBoxLayout()
        exportAllButton: QPushButton = QPushButton("一键导出所有数据")
        exportAllButton.clicked.connect(self.OnExportAllDataClicked)
        exportAllButton.setEnabled(False)
        exportAllButton.setFixedSize(200, 60)
        exportAllButtonLayout.addStretch()
        exportAllButtonLayout.addWidget(exportAllButton)
        exportAllButtonLayout.addStretch()

        controlLayout.addLayout(exportAllButtonLayout)

        self.mainLayout.addTab(controlWidget, "控制台")

    def CreateTrajectoryTab(self) -> None:
        """创建拟合轨迹Tab内容"""
        trajectoryWidget: QWidget = QWidget()
        trajectoryLayout: QVBoxLayout = QVBoxLayout(trajectoryWidget)
        imageLayout: QHBoxLayout = QHBoxLayout()

        detectionGroup: QGroupBox = QGroupBox("检测画面")
        detectionLayout: QVBoxLayout = QVBoxLayout()
        self.detectionLabel: QLabel = QLabel()
        self.detectionLabel.setFixedSize(640, 360)
        self.detectionLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detectionLayout.addWidget(self.detectionLabel)
        detectionGroup.setLayout(detectionLayout)

        statisticsGroup: QGroupBox = QGroupBox("拟合轨迹")
        statisticsLayout: QVBoxLayout = QVBoxLayout()
        self.statisticsLabel: QLabel = QLabel()
        self.statisticsLabel.setFixedSize(640, 420)
        self.statisticsLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        statisticsLayout.addWidget(self.statisticsLabel)
        statisticsGroup.setLayout(statisticsLayout)

        imageLayout.addWidget(detectionGroup)
        imageLayout.addWidget(statisticsGroup)

        frameControlLayout: QHBoxLayout = QHBoxLayout()

        self.frameSlider: QSlider = QSlider(Qt.Orientation.Horizontal)
        self.frameSlider.setMinimum(0)
        # 使用当前 result 设置最大值（如果没有 result，暂时设为 0）
        maxFrame: int = self.result.totalFrame - 1 if self.result else 0
        self.frameSlider.setMaximum(maxFrame)
        self.frameSlider.setValue(0)
        self.frameSlider.valueChanged.connect(self.OnFrameSliderChanged)

        self.frameLabel: QLabel = QLabel(f"帧: 0 / {maxFrame}")
        self.frameLabel.setFixedWidth(150)

        frameControlLayout.addWidget(QLabel("帧选择:"))
        frameControlLayout.addWidget(self.frameSlider)
        frameControlLayout.addWidget(self.frameLabel)

        trajectoryLayout.addLayout(imageLayout)
        trajectoryLayout.addLayout(frameControlLayout)

        self.mainLayout.addTab(trajectoryWidget, "拟合轨迹")

    def CreateDataTab(self) -> None:
        """创建数据Tab内容"""

        def GetBarChartView(chartTitle: str, valueList: list[float]) -> QChartView:
            """创建柱状图视图"""
            barSetObject: QBarSet = QBarSet(chartTitle)
            barSetObject.append(valueList)
            barSeriesObject: QBarSeries = QBarSeries()
            barSeriesObject.append(barSetObject)

            chartObject: QChart = QChart()
            chartObject.addSeries(barSeriesObject)
            chartObject.setTitle(chartTitle)
            chartObject.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

            axisXObject: QBarCategoryAxis = QBarCategoryAxis()
            chartObject.addAxis(axisXObject, Qt.AlignmentFlag.AlignBottom)
            axisYObject: QValueAxis = QValueAxis()
            chartObject.addAxis(axisYObject, Qt.AlignmentFlag.AlignLeft)
            barSeriesObject.attachAxis(axisYObject)

            chartObject.legend().setVisible(False)
            chartObject.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)
            chartObject.setMargins(QMargins(0, 0, 0, 0))
            barSeriesObject.setBarWidth(1.0)

            chartViewObject: QChartView = QChartView(chartObject)
            chartViewObject.setRenderHint(QPainter.RenderHint.Antialiasing)
            return chartViewObject

        def GetTableView(dataFrame: pd.DataFrame) -> QTableView:
            """创建表格视图"""
            pandasModelObject: PandasModel = PandasModel(dataFrame)
            tableViewObject: QTableView = QTableView()
            tableViewObject.setModel(pandasModelObject)
            return tableViewObject

        def GetDataViewWidget(dataFrame: pd.DataFrame, columnList: list[str]) -> QWidget:
            """创建包含柱状图和表格的数据展示组件"""
            chartViewLayoutObject: QHBoxLayout = QHBoxLayout()
            for columnName in columnList:
                columnValueList: list[float] = dataFrame[columnName].tolist()
                barChartViewObject: QChartView = GetBarChartView(columnName, columnValueList)
                chartViewLayoutObject.addWidget(barChartViewObject)

            tableViewObject: QTableView = GetTableView(dataFrame)
            tableViewObject.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            flexLayoutObject: QVBoxLayout = QVBoxLayout()
            flexLayoutObject.addLayout(chartViewLayoutObject)
            flexLayoutObject.addWidget(tableViewObject)
            widgetObject: QWidget = QWidget()
            widgetObject.setLayout(flexLayoutObject)
            return widgetObject

        for laneIndex in range(len(self.result.laneList)):
            laneObject: Lane = self.result.laneList[laneIndex]
            tabName: str = laneObject.sectionName

            mainTabWidget: QTabWidget = QTabWidget()

            timeDataFrame: pd.DataFrame = laneObject.GetTimeOccupancyDataFrame()
            timeWidget: QWidget = GetDataViewWidget(timeDataFrame, ["时间占有率(%)"])
            mainTabWidget.addTab(timeWidget, "时间占有率")

            spaceDataFrame: pd.DataFrame = laneObject.GetSpaceOccupancyDataFrame()
            spaceWidget: QWidget = GetDataViewWidget(spaceDataFrame, ["空间占有率(%)"])
            mainTabWidget.addTab(spaceWidget, "空间占有率")

            flowDataFrame: pd.DataFrame = laneObject.GetFlowDataFrame()
            flowWidget: QWidget = GetDataViewWidget(flowDataFrame, ["流量(辆/分钟)"])
            mainTabWidget.addTab(flowWidget, "流量")

            queueDataFrame: pd.DataFrame = laneObject.GetQueueLengthDataFrame()
            queueWidget: QWidget = GetDataViewWidget(queueDataFrame, ["排队长度(m)"])
            mainTabWidget.addTab(queueWidget, "排队长度")

            sectionDataFrame: pd.DataFrame = laneObject.GetSectionPassingDataFrame()
            sectionWidget: QWidget = GetDataViewWidget(sectionDataFrame, ["速度(m/s)", "车头时距(s)", "间距(m)"])
            mainTabWidget.addTab(sectionWidget, "过车断面")

            self.mainLayout.addTab(mainTabWidget, tabName)

    def OnFrameSliderChanged(self, value: int) -> None:
        """帧滑动条值变化时的回调"""
        self.currentFrameIndex = value
        if self.result is not None:
            totalFrames: int = self.result.totalFrame
            self.frameLabel.setText(f"帧: {value} / {totalFrames - 1}")
            self.UpdateImages(value)

    def UpdateImages(self, frameIndex: int) -> None:
        """更新两个图像显示"""
        if self.result is None:
            return

        try:
            detectionImage: np.ndarray = self.result.GetDetectionFrameImage(
                frameIndex=frameIndex,
                targetWidth=640,
                targetHeight=360
            )

            statisticsImage: np.ndarray = self.result.GetStatisticsFrameImage(
                frameIndex=frameIndex,
                targetWidth=640,
                targetHeight=420
            )

            detectionImageRGB: np.ndarray = cv2.cvtColor(detectionImage, cv2.COLOR_BGR2RGB)
            statisticsImageRGB: np.ndarray = cv2.cvtColor(statisticsImage, cv2.COLOR_BGR2RGB)

            detectionHeight, detectionWidth, detectionChannel = detectionImageRGB.shape
            statisticsHeight, statisticsWidth, statisticsChannel = statisticsImageRGB.shape

            bytesPerLineDetection: int = detectionChannel * detectionWidth
            bytesPerLineStatistics: int = statisticsChannel * statisticsWidth

            detectionQImage: QImage = QImage(
                detectionImageRGB.data,
                detectionWidth,
                detectionHeight,
                bytesPerLineDetection,
                QImage.Format.Format_RGB888
            )

            statisticsQImage: QImage = QImage(
                statisticsImageRGB.data,
                statisticsWidth,
                statisticsHeight,
                bytesPerLineStatistics,
                QImage.Format.Format_RGB888
            )

            self.detectionLabel.setPixmap(QPixmap.fromImage(detectionQImage))
            self.statisticsLabel.setPixmap(QPixmap.fromImage(statisticsQImage))

        except Exception as exception:
            print(f"更新图像失败: {str(exception)}")

    def OnExportAllDataClicked(self) -> None:
        """点击一键导出所有数据按钮"""
        if self.result is None:
            QMessageBox.warning(self, "警告", "没有可导出的数据")
            return

        exportDirectoryPath: str = QFileDialog.getExistingDirectory(self, "选择导出目录", "./Data")
        if not exportDirectoryPath:
            return

        try:
            self.result.ExportAllDataToDirectory(exportDirectoryPath)
            QMessageBox.information(self, "成功", f"数据已成功导出到:\n{exportDirectoryPath}")
        except Exception as exception:
            QMessageBox.critical(self, "错误", f"导出失败: {str(exception)}")
