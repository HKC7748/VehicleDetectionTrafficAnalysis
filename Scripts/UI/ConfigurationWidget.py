import numpy as np
from typing import (Optional,
                    List,
                    Dict)
from PySide6.QtWidgets import (QWidget,
                               QVBoxLayout,
                               QHBoxLayout,
                               QFormLayout,
                               QGroupBox,
                               QPushButton,
                               QLabel,
                               QLineEdit,
                               QListWidget,
                               QFileDialog,
                               QMessageBox,
                               QTableWidget,
                               QTableWidgetItem,
                               QHeaderView,
                               QSpinBox,
                               QSplitter)
from PySide6.QtCore import Qt
from Scripts.DetectionPlan import DetectionPlan
from Scripts.Lane import Lane


def PopulatePointTable(tableWidget: QTableWidget, points: Optional[np.ndarray]) -> None:
    """填充点表格"""
    tableWidget.blockSignals(True)
    tableWidget.setRowCount(0)
    if points is not None and len(points.shape) == 2 and points.shape[1] == 2:
        for rowIndex in range(points.shape[0]):
            tableWidget.insertRow(rowIndex)
            for columnIndex in range(2):
                tableWidget.setItem(rowIndex, columnIndex, QTableWidgetItem(str(points[rowIndex, columnIndex])))
    tableWidget.blockSignals(False)


def ReadNumpyArrayFromTableWidget(tableWidget: QTableWidget) -> Optional[np.ndarray]:
    """从QTableWidget读取数据并转换为numpy数组"""
    rowCount: int = tableWidget.rowCount()
    columnCount: int = tableWidget.columnCount()

    if rowCount == 0 or columnCount == 0:
        return None

    pointsList: List[List[float]] = []

    for rowIndex in range(rowCount):
        rowData: List[float] = []
        for columnIndex in range(columnCount):
            item: QTableWidgetItem = tableWidget.item(rowIndex, columnIndex)
            if item is None or not item.text().strip():
                return None
            try:
                rowData.append(float(item.text()))
            except ValueError:
                return None
        pointsList.append(rowData)

    if not pointsList:
        return None

    return np.array(pointsList, dtype=np.float32)


def GetSelectedRowFromTableWidget(tableWidget: QTableWidget) -> int:
    """获取QTableWidget的选中行"""
    return tableWidget.currentRow()


def GetRowCountFromTableWidget(tableWidget: QTableWidget) -> int:
    """获取QTableWidget的行数"""
    return tableWidget.rowCount()


def InsertRowIntoTableWidget(tableWidget: QTableWidget, defaultValues: List[str]) -> None:
    """向QTableWidget插入一行"""
    rowPosition: int = tableWidget.rowCount()
    tableWidget.insertRow(rowPosition)
    for columnIndex, value in enumerate(defaultValues):
        tableWidget.setItem(rowPosition, columnIndex, QTableWidgetItem(value))


def RemoveSelectedRowFromTableWidget(tableWidget: QTableWidget) -> None:
    """从QTableWidget删除选中行"""
    currentRow: int = tableWidget.currentRow()
    if currentRow >= 0:
        tableWidget.removeRow(currentRow)


def UpdateLaneFromTable(laneObject: Lane, linePointsTable: QTableWidget, edgePointsTable: QTableWidget) -> None:
    """从表格更新车道点数据"""
    linePointsArray: Optional[np.ndarray] = ReadNumpyArrayFromTableWidget(linePointsTable)
    edgePointsArray: Optional[np.ndarray] = ReadNumpyArrayFromTableWidget(edgePointsTable)

    if linePointsArray is not None and linePointsArray.shape == (2, 2):
        laneObject.linePoints = linePointsArray

    if edgePointsArray is not None and edgePointsArray.shape[1] == 2:
        laneObject.edgePoints = edgePointsArray


def ReadVehicleDataFromTable(vehicleTable: QTableWidget) -> Dict[str, Dict[str, float]]:
    """从车辆表格读取数据"""
    vehiclesDictionary: Dict[str, Dict[str, float]] = {}

    for rowIndex in range(vehicleTable.rowCount()):
        typeItem: QTableWidgetItem = vehicleTable.item(rowIndex, 0)
        lengthItem: QTableWidgetItem = vehicleTable.item(rowIndex, 1)
        widthItem: QTableWidgetItem = vehicleTable.item(rowIndex, 2)

        if typeItem and lengthItem and widthItem:
            vehicleType: str = typeItem.text().strip()
            try:
                lengthValue: float = float(lengthItem.text())
                widthValue: float = float(widthItem.text())
                vehiclesDictionary[vehicleType] = {"length": lengthValue, "width": widthValue}
            except ValueError:
                continue

    return vehiclesDictionary


class ConfigurationWidget(QWidget):
    """配置组件 - 用于编辑检测计划参数"""

    def __init__(self, detectionPlan: DetectionPlan) -> None:
        super().__init__()
        self.detectionPlan: DetectionPlan = detectionPlan
        self.mainLayout: QHBoxLayout = QHBoxLayout(self)
        basicGroup: QGroupBox = QGroupBox("基本参数")
        basicLayout: QFormLayout = QFormLayout()
        fileLayout: QHBoxLayout = QHBoxLayout()

        self.importButton: QPushButton = QPushButton("导入配置")
        self.importButton.clicked.connect(self.OnImportClicked)

        self.exportButton: QPushButton = QPushButton("导出配置")
        self.exportButton.clicked.connect(self.OnExportClicked)
        basicLayout.addRow("配置:", fileLayout)

        nameLayout: QHBoxLayout = QHBoxLayout()
        self.nameInput: QLineEdit = QLineEdit()
        self.nameInput.setPlaceholderText("输入配置名称...")
        self.nameInput.textChanged.connect(self.OnNameChanged)
        nameLayout.addWidget(self.nameInput)
        basicLayout.addRow("配置名称:", nameLayout)

        fileLayout.addWidget(self.importButton)
        fileLayout.addWidget(self.exportButton)
        fileLayout.addStretch()

        yoloPathLayout: QHBoxLayout = QHBoxLayout()
        self.yoloModelPathInput: QLineEdit = QLineEdit()
        self.yoloModelPathInput.setPlaceholderText("选择 YOLO 模型文件...")
        self.yoloModelPathInput.textChanged.connect(self.OnYoloModelPathChanged)
        yoloBrowseButton: QPushButton = QPushButton("浏览")
        yoloBrowseButton.clicked.connect(self.BrowseModelPath)
        yoloPathLayout.addWidget(self.yoloModelPathInput)
        yoloPathLayout.addWidget(yoloBrowseButton)
        basicLayout.addRow("YOLO 模型路径:", yoloPathLayout)

        videoPathLayout: QHBoxLayout = QHBoxLayout()
        self.videoSourceInput: QLineEdit = QLineEdit()
        self.videoSourceInput.setPlaceholderText("选择视频文件...")
        self.videoSourceInput.textChanged.connect(self.OnVideoSourceChanged)
        videoBrowseButton: QPushButton = QPushButton("浏览")
        videoBrowseButton.clicked.connect(self.BrowseVideoPath)
        videoPathLayout.addWidget(self.videoSourceInput)
        videoPathLayout.addWidget(videoBrowseButton)
        basicLayout.addRow("视频源:", videoPathLayout)

        self.timeOccupancyWindowsInput: QSpinBox = QSpinBox()
        self.timeOccupancyWindowsInput.setRange(1, 99999)
        self.timeOccupancyWindowsInput.valueChanged.connect(self.OnTimeOccupancyWindowsChanged)
        basicLayout.addRow("时间占有率窗口:", self.timeOccupancyWindowsInput)

        self.spaceOccupancyWindowsInput: QSpinBox = QSpinBox()
        self.spaceOccupancyWindowsInput.setRange(1, 99999)
        self.spaceOccupancyWindowsInput.valueChanged.connect(self.OnSpaceOccupancyWindowsChanged)
        basicLayout.addRow("空间占有率窗口:", self.spaceOccupancyWindowsInput)

        self.flowWindowsInput: QSpinBox = QSpinBox()
        self.flowWindowsInput.setRange(1, 99999)
        self.flowWindowsInput.valueChanged.connect(self.OnFlowWindowsChanged)
        basicLayout.addRow("流量统计窗口:", self.flowWindowsInput)

        self.queueLengthWindowsInput: QSpinBox = QSpinBox()
        self.queueLengthWindowsInput.setRange(1, 99999)
        self.queueLengthWindowsInput.valueChanged.connect(self.OnQueueLengthWindowsChanged)
        basicLayout.addRow("排队长度窗口:", self.queueLengthWindowsInput)

        basicGroup.setLayout(basicLayout)
        self.mainLayout.addWidget(basicGroup)

        vehicleGroup: QGroupBox = QGroupBox("车辆类型")
        vehicleLayout: QVBoxLayout = QVBoxLayout()
        self.vehicleTable: QTableWidget = QTableWidget(0, 3)
        self.vehicleTable.setHorizontalHeaderLabels(["类型", "长度(m)", "宽度(m)"])
        self.vehicleTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.vehicleTable.cellChanged.connect(self.OnVehicleTableChanged)

        vehicleButtonLayout: QHBoxLayout = QHBoxLayout()
        addVehicleButton: QPushButton = QPushButton("添加")
        addVehicleButton.clicked.connect(self.OnAddVehicleClicked)
        removeVehicleButton: QPushButton = QPushButton("删除")
        removeVehicleButton.clicked.connect(self.onRemoveVehicleClicked)
        vehicleButtonLayout.addWidget(addVehicleButton)
        vehicleButtonLayout.addWidget(removeVehicleButton)
        vehicleButtonLayout.addStretch()

        vehicleLayout.addWidget(self.vehicleTable)
        vehicleLayout.addLayout(vehicleButtonLayout)
        vehicleGroup.setLayout(vehicleLayout)
        self.mainLayout.addWidget(vehicleGroup)

        pointGroup: QGroupBox = QGroupBox("透视/平行点")
        pointLayout: QVBoxLayout = QVBoxLayout()

        perspectiveLabel: QLabel = QLabel("透视点 (Nx2)")
        self.perspectiveTable: QTableWidget = QTableWidget(0, 2)
        self.perspectiveTable.setHorizontalHeaderLabels(["X", "Y"])
        self.perspectiveTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.perspectiveTable.cellChanged.connect(self.OnPerspectiveTableChanged)

        perspectiveButtonLayout: QHBoxLayout = QHBoxLayout()
        addPerspectivePointButton: QPushButton = QPushButton("添加")
        addPerspectivePointButton.clicked.connect(self.onAddPerspectivePointClicked)
        removePerspectivePointButton: QPushButton = QPushButton("删除")
        removePerspectivePointButton.clicked.connect(self.OnRemovePerspectivePointClicked)
        perspectiveButtonLayout.addWidget(addPerspectivePointButton)
        perspectiveButtonLayout.addWidget(removePerspectivePointButton)
        perspectiveButtonLayout.addStretch()

        pointLayout.addWidget(perspectiveLabel)
        pointLayout.addWidget(self.perspectiveTable)
        pointLayout.addLayout(perspectiveButtonLayout)

        parallelLabel: QLabel = QLabel("平行点 (Nx2)")
        self.parallelTable: QTableWidget = QTableWidget(0, 2)
        self.parallelTable.setHorizontalHeaderLabels(["X", "Y"])
        self.parallelTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parallelTable.cellChanged.connect(self.OnParallelTableChanged)

        parallelButtonLayout: QHBoxLayout = QHBoxLayout()
        addParallelPointButton: QPushButton = QPushButton("添加")
        addParallelPointButton.clicked.connect(self.OnAddParallelPointClicked)
        removeParallelPointButton: QPushButton = QPushButton("删除")
        removeParallelPointButton.clicked.connect(self.OnRemoveParallelPointClicked)
        parallelButtonLayout.addWidget(addParallelPointButton)
        parallelButtonLayout.addWidget(removeParallelPointButton)
        parallelButtonLayout.addStretch()

        pointLayout.addWidget(parallelLabel)
        pointLayout.addWidget(self.parallelTable)
        pointLayout.addLayout(parallelButtonLayout)

        pointGroup.setLayout(pointLayout)
        self.mainLayout.addWidget(pointGroup)

        laneGroup: QGroupBox = QGroupBox("车道编辑")
        laneLayout: QHBoxLayout = QHBoxLayout()

        splitterWidget: QSplitter = QSplitter(Qt.Orientation.Horizontal)

        self.laneListWidget: QListWidget = QListWidget()
        self.laneListWidget.currentRowChanged.connect(self.OnLaneSelectionChanged)

        laneListButtons: QHBoxLayout = QHBoxLayout()
        addLaneButton: QPushButton = QPushButton("添加")
        addLaneButton.clicked.connect(self.OnAddLaneClicked)
        removeLaneButton: QPushButton = QPushButton("删除")
        removeLaneButton.clicked.connect(self.OnRemoveLaneClicked)
        laneListButtons.addWidget(addLaneButton)
        laneListButtons.addWidget(removeLaneButton)
        laneListButtons.addStretch()

        rightWidget: QWidget = QWidget()
        rightLayout: QFormLayout = QFormLayout(rightWidget)
        rightLayout.setContentsMargins(0, 0, 0, 0)

        rightLayout.addWidget(QLabel("车道列表"))
        rightLayout.addWidget(self.laneListWidget)
        rightLayout.addRow(laneListButtons)

        self.laneSectionNameInput: QLineEdit = QLineEdit()
        self.laneSectionNameInput.textChanged.connect(self.OnLaneSectionNameChanged)
        laneSectionNameLayout: QHBoxLayout = QHBoxLayout()
        laneSectionNameLayout.addWidget(QLabel("名称"))
        laneSectionNameLayout.addWidget(self.laneSectionNameInput)
        rightLayout.addRow(laneSectionNameLayout)

        linePointsLabel: QLabel = QLabel("检测线点 (2x2)")
        self.laneLinePointsTable: QTableWidget = QTableWidget(2, 2)
        self.laneLinePointsTable.setHorizontalHeaderLabels(["X", "Y"])
        self.laneLinePointsTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.laneLinePointsTable.cellChanged.connect(self.OnLaneLinePointsChanged)
        rightLayout.addRow(linePointsLabel)
        rightLayout.addRow(self.laneLinePointsTable)

        edgePointsLabel: QLabel = QLabel("边界点 (Nx2)")
        self.laneEdgePointsTable: QTableWidget = QTableWidget(0, 2)
        self.laneEdgePointsTable.setHorizontalHeaderLabels(["X", "Y"])
        self.laneEdgePointsTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.laneEdgePointsTable.cellChanged.connect(self.OnLaneEdgePointsChanged)
        rightLayout.addRow(edgePointsLabel)
        rightLayout.addRow(self.laneEdgePointsTable)

        edgePointsButtons: QHBoxLayout = QHBoxLayout()
        addEdgePointButton: QPushButton = QPushButton("添加行")
        addEdgePointButton.clicked.connect(self.OnAddEdgePointClicked)
        removeEdgePointButton: QPushButton = QPushButton("删除行")
        removeEdgePointButton.clicked.connect(self.OnRemoveEdgePointClicked)
        edgePointsButtons.addWidget(addEdgePointButton)
        edgePointsButtons.addWidget(removeEdgePointButton)
        edgePointsButtons.addStretch()
        rightLayout.addRow("", edgePointsButtons)

        splitterWidget.addWidget(rightWidget)
        splitterWidget.setStretchFactor(0, 1)
        splitterWidget.setStretchFactor(1, 2)

        laneLayout.addWidget(splitterWidget)
        laneGroup.setLayout(laneLayout)
        self.mainLayout.addWidget(laneGroup)

    def BrowseModelPath(self) -> None:
        """浏览YOLO模型文件"""
        filePath: str
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "./Model",
            "模型文件 (*.pt *.onnx)"
        )
        if filePath:
            self.yoloModelPathInput.setText(filePath)

    def BrowseVideoPath(self) -> None:
        """浏览视频文件"""
        filePath: str
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "./Resource",
            "视频文件 (*.mp4 *.avi *.mov)"
        )
        if filePath:
            self.videoSourceInput.setText(filePath)

    def OnImportClicked(self) -> None:
        """处理导入配置按钮点击事件"""
        filePath: str
        filePath, _ = QFileDialog.getOpenFileName(self, "导入配置", "./Configure", "JSON 文件 (*.json)")
        if not filePath:
            return
        try:
            self.detectionPlan.FromJson(filePath)
            self.PopulateFromDetectionPlan()
            QMessageBox.information(self, "成功", "配置导入成功")
        except Exception as exception:
            QMessageBox.critical(self, "错误", f"导入失败: {str(exception)}")

    def OnExportClicked(self) -> None:
        """处理导出配置按钮点击事件"""
        filePath: str
        filePath, _ = QFileDialog.getSaveFileName(self, "导出配置", "./Configure", "JSON 文件 (*.json)")
        if not filePath:
            return
        try:
            self.detectionPlan.ToJson(filePath)
            QMessageBox.information(self, "成功", "配置导出成功")
        except Exception as exception:
            QMessageBox.critical(self, "错误", f"导出失败: {str(exception)}")

    def OnNameChanged(self, text: str) -> None:
        """处理配置名称更改"""
        self.detectionPlan.name = text

    def OnYoloModelPathChanged(self, text: str) -> None:
        """处理YOLO模型路径更改"""
        self.detectionPlan.yoloModelPath = text if text else None

    def OnVideoSourceChanged(self, text: str) -> None:
        """处理视频源更改"""
        self.detectionPlan.videoSource = text if text else None

    def OnTimeOccupancyWindowsChanged(self, value: int) -> None:
        """处理时间占有率窗口更改"""
        self.detectionPlan.timeOccupancyWindows = value
        for lane in self.detectionPlan.laneList:
            lane.timeOccupancyWindows = value

    def OnSpaceOccupancyWindowsChanged(self, value: int) -> None:
        """处理空间占有率窗口更改"""
        self.detectionPlan.spaceOccupancyWindows = value
        for lane in self.detectionPlan.laneList:
            lane.spaceOccupancyWindows = value

    def OnFlowWindowsChanged(self, value: int) -> None:
        """处理流量统计窗口更改"""
        self.detectionPlan.flowWindows = value
        for lane in self.detectionPlan.laneList:
            lane.flowWindows = value

    def OnQueueLengthWindowsChanged(self, value: int) -> None:
        """处理排队长度窗口更改"""
        self.detectionPlan.queueLengthWindows = value
        for lane in self.detectionPlan.laneList:
            lane.queueLengthWindows = value

    def OnVehicleTableChanged(self) -> None:
        """处理车辆表格更改"""
        self.SyncVehiclesFromTable()

    def onAddPerspectivePointClicked(self) -> None:
        """添加透视点行"""
        self.perspectiveTable.blockSignals(True)
        InsertRowIntoTableWidget(self.perspectiveTable, ["0", "0"])
        self.perspectiveTable.blockSignals(False)
        self.OnPerspectiveTableChanged()

    def OnRemovePerspectivePointClicked(self) -> None:
        """删除选中的透视点行"""
        self.perspectiveTable.blockSignals(True)
        RemoveSelectedRowFromTableWidget(self.perspectiveTable)
        self.perspectiveTable.blockSignals(False)
        self.OnPerspectiveTableChanged()

    def OnAddParallelPointClicked(self) -> None:
        """添加平行点行"""
        self.parallelTable.blockSignals(True)
        InsertRowIntoTableWidget(self.parallelTable, ["0", "0"])
        self.parallelTable.blockSignals(False)
        self.OnParallelTableChanged()

    def OnRemoveParallelPointClicked(self) -> None:
        """删除选中的平行点行"""
        self.parallelTable.blockSignals(True)
        RemoveSelectedRowFromTableWidget(self.parallelTable)
        self.parallelTable.blockSignals(False)
        self.OnParallelTableChanged()

    def OnAddVehicleClicked(self) -> None:
        """添加车辆行"""
        self.vehicleTable.blockSignals(True)
        InsertRowIntoTableWidget(self.vehicleTable, ["", "", ""])
        self.vehicleTable.blockSignals(False)
        self.SyncVehiclesFromTable()

    def onRemoveVehicleClicked(self) -> None:
        """删除选中的车辆行"""
        self.vehicleTable.blockSignals(True)
        RemoveSelectedRowFromTableWidget(self.vehicleTable)
        self.vehicleTable.blockSignals(False)
        self.SyncVehiclesFromTable()

    def SyncVehiclesFromTable(self) -> None:
        """从表格同步车辆数据"""
        vehiclesDictionary: Dict[str, Dict[str, float]] = ReadVehicleDataFromTable(self.vehicleTable)
        self.detectionPlan.vehicles = vehiclesDictionary if vehiclesDictionary else None

    def OnPerspectiveTableChanged(self) -> None:
        """处理透视点表格更改"""
        self.SyncPointsFromTable(self.perspectiveTable, "perspective")

    def OnParallelTableChanged(self) -> None:
        """处理平行点表格更改"""
        self.SyncPointsFromTable(self.parallelTable, "parallel")

    def SyncPointsFromTable(self, tableWidget: QTableWidget, pointType: str) -> None:
        """从表格同步点数据"""
        pointsArray: Optional[np.ndarray] = ReadNumpyArrayFromTableWidget(tableWidget)

        if pointsArray is None or pointsArray.shape[1] != 2:
            return

        if pointType == "perspective":
            self.detectionPlan.pointsPerspective = pointsArray
        else:
            self.detectionPlan.pointsParallel = pointsArray

    def OnLaneSelectionChanged(self, currentRow: int) -> None:
        """处理车道选择更改"""
        if currentRow < 0 or currentRow >= len(self.detectionPlan.laneList):
            self.ClearLaneDetail()
            return
        laneObject: Lane = self.detectionPlan.laneList[currentRow]
        self.PopulateLaneDetail(laneObject)

    def ClearLaneDetail(self) -> None:
        """清除车道详情"""
        self.laneSectionNameInput.blockSignals(True)
        self.laneLinePointsTable.blockSignals(True)
        self.laneEdgePointsTable.blockSignals(True)
        self.laneSectionNameInput.clear()
        self.laneLinePointsTable.setRowCount(0)
        self.laneEdgePointsTable.setRowCount(0)
        self.laneSectionNameInput.blockSignals(False)
        self.laneLinePointsTable.blockSignals(False)
        self.laneEdgePointsTable.blockSignals(False)

    def PopulateLaneDetail(self, laneObject: Lane) -> None:
        """填充车道详情"""
        self.laneSectionNameInput.blockSignals(True)
        self.laneLinePointsTable.blockSignals(True)
        self.laneEdgePointsTable.blockSignals(True)

        self.laneSectionNameInput.setText(laneObject.sectionName)

        self.laneLinePointsTable.setRowCount(2)
        for rowIndex in range(2):
            for columnIndex in range(2):
                self.laneLinePointsTable.setItem(rowIndex, columnIndex, QTableWidgetItem(str(laneObject.linePoints[rowIndex, columnIndex])))

        edgeRowsCount: int = laneObject.edgePoints.shape[0]
        self.laneEdgePointsTable.setRowCount(edgeRowsCount)
        for rowIndex in range(edgeRowsCount):
            for columnIndex in range(2):
                self.laneEdgePointsTable.setItem(rowIndex, columnIndex, QTableWidgetItem(str(laneObject.edgePoints[rowIndex, columnIndex])))

        self.laneSectionNameInput.blockSignals(False)
        self.laneLinePointsTable.blockSignals(False)
        self.laneEdgePointsTable.blockSignals(False)

    def OnLaneSectionNameChanged(self, text: str) -> None:
        """处理车道名称更改"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        laneObject: Lane = self.detectionPlan.laneList[currentRow]
        laneObject.sectionName = text
        self.laneListWidget.currentItem().setText(text)

    def OnLaneLinePointsChanged(self) -> None:
        """处理检测线点表格更改"""
        self.SyncLaneLinePointsFromTable()

    def SyncLaneLinePointsFromTable(self) -> None:
        """从表格同步检测线点数据"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        laneObject: Lane = self.detectionPlan.laneList[currentRow]
        UpdateLaneFromTable(laneObject, self.laneLinePointsTable, self.laneEdgePointsTable)

    def OnLaneEdgePointsChanged(self) -> None:
        """处理边界点表格更改"""
        self.SyncLaneEdgePointsFromTable()

    def SyncLaneEdgePointsFromTable(self) -> None:
        """从表格同步边界点数据"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        laneObject: Lane = self.detectionPlan.laneList[currentRow]
        UpdateLaneFromTable(laneObject, self.laneLinePointsTable, self.laneEdgePointsTable)

    def OnAddEdgePointClicked(self) -> None:
        """添加边界点行"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        self.laneEdgePointsTable.blockSignals(True)
        InsertRowIntoTableWidget(self.laneEdgePointsTable, ["0", "0"])
        self.laneEdgePointsTable.blockSignals(False)
        self.SyncLaneEdgePointsFromTable()

    def OnRemoveEdgePointClicked(self) -> None:
        """删除选中的边界点行"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        self.laneEdgePointsTable.blockSignals(True)
        RemoveSelectedRowFromTableWidget(self.laneEdgePointsTable)
        self.laneEdgePointsTable.blockSignals(False)
        self.SyncLaneEdgePointsFromTable()

    def OnAddLaneClicked(self) -> None:
        """添加车道"""
        newLane: Lane = Lane(
            sectionName="new_lane",
            linePoints=np.array([[0, 0], [100, 0]], dtype=np.float32),
            edgePoints=np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float32),
            vehicles=self.detectionPlan.vehicles or {},
            timeOccupancyWindows=self.detectionPlan.timeOccupancyWindows,
            spaceOccupancyWindows=self.detectionPlan.spaceOccupancyWindows,
            flowWindows=self.detectionPlan.flowWindows,
            queueLengthWindows=self.detectionPlan.queueLengthWindows
        )
        self.detectionPlan.laneList.append(newLane)
        self.laneListWidget.addItem(newLane.sectionName)
        self.laneListWidget.setCurrentRow(self.laneListWidget.count() - 1)

    def OnRemoveLaneClicked(self) -> None:
        """删除车道"""
        currentRow: int = self.laneListWidget.currentRow()
        if currentRow < 0:
            return
        self.detectionPlan.laneList.pop(currentRow)
        self.laneListWidget.takeItem(currentRow)
        if self.laneListWidget.count() > 0:
            self.laneListWidget.setCurrentRow(min(currentRow, self.laneListWidget.count() - 1))
        else:
            self.ClearLaneDetail()

    def PopulateFromDetectionPlan(self) -> None:
        """从检测计划填充控件"""
        detectionPlanObject: DetectionPlan = self.detectionPlan

        self.nameInput.setText(self.detectionPlan.name or "")
        self.yoloModelPathInput.setText(detectionPlanObject.yoloModelPath or "")
        self.videoSourceInput.setText(detectionPlanObject.videoSource or "")
        self.timeOccupancyWindowsInput.setValue(detectionPlanObject.timeOccupancyWindows)
        self.spaceOccupancyWindowsInput.setValue(detectionPlanObject.spaceOccupancyWindows)
        self.flowWindowsInput.setValue(detectionPlanObject.flowWindows)
        self.queueLengthWindowsInput.setValue(detectionPlanObject.queueLengthWindows)

        self.PopulateVehicleTable(detectionPlanObject.vehicles)

        PopulatePointTable(self.perspectiveTable, detectionPlanObject.pointsPerspective)
        PopulatePointTable(self.parallelTable, detectionPlanObject.pointsParallel)

        self.PopulateLaneList(detectionPlanObject.laneList)

    def PopulateVehicleTable(self, vehicles: Optional[Dict[str, Dict[str, float]]]) -> None:
        """填充车辆表格"""
        self.vehicleTable.blockSignals(True)
        self.vehicleTable.setRowCount(0)
        if vehicles:
            for vehicleType, dimensions in vehicles.items():
                rowPosition: int = self.vehicleTable.rowCount()
                self.vehicleTable.insertRow(rowPosition)
                self.vehicleTable.setItem(rowPosition, 0, QTableWidgetItem(vehicleType))
                self.vehicleTable.setItem(rowPosition, 1, QTableWidgetItem(str(dimensions["length"])))
                self.vehicleTable.setItem(rowPosition, 2, QTableWidgetItem(str(dimensions["width"])))
        self.vehicleTable.blockSignals(False)

    def PopulateLaneList(self, laneList: List[Lane]) -> None:
        """填充车道列表"""
        self.laneListWidget.blockSignals(True)
        self.laneListWidget.clear()
        for lane in laneList:
            self.laneListWidget.addItem(lane.sectionName)
        if self.laneListWidget.count() > 0:
            self.laneListWidget.setCurrentRow(0)
        self.laneListWidget.blockSignals(False)
