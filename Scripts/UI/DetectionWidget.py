from typing import Optional
from PySide6.QtCore import (Signal,
                            QThread)
from PySide6.QtWidgets import (QWidget,
                               QVBoxLayout,
                               QHBoxLayout,
                               QPushButton,
                               QProgressBar,
                               QTextEdit,
                               QMessageBox)
from ultralytics.engine.results import (Boxes,
                                        Results)

from Scripts.DetectionPlan import DetectionPlan
from Scripts.Result import Result
from Scripts.VehicleDetectionSystem import VehicleDetectionSystem
from Scripts.UI.VisualizationWidget import VisualizationWidget


class DetectionWorker(QThread):
    """检测工作线程，负责在后台执行车辆检测任务"""
    detectionOutputSignal: Signal = Signal(str)
    detectionFinishedSignal: Signal = Signal(bool, str)

    def __init__(self, detectionSystem: VehicleDetectionSystem) -> None:
        super().__init__()
        self.detectionSystem: VehicleDetectionSystem = detectionSystem
        self.detectionPlan: DetectionPlan = self.detectionSystem.detectionPlan
        self.isCancelled: bool = False

    def run(self) -> None:
        """执行检测任务的主要方法"""
        try:
            self.detectionOutputSignal.emit("开始检测...\n")
            self.detectionSystem.StartDetection(progressCallback=self.ProgressCallback)
            if self.isCancelled:
                self.detectionFinishedSignal.emit(False, "检测已取消")
                return
            self.detectionFinishedSignal.emit(True, "检测完成")
        except Exception as exception:
            self.detectionFinishedSignal.emit(False, f"检测失败: {str(exception)}")

    def ProgressCallback(self, frameIndex: int, totalFrames: int, result: Results) -> None:
        """处理检测进度回调"""
        if self.isCancelled:
            return
        boxes: Boxes = result.boxes
        classNames = result.names
        classCountsDictionary: dict[str, int] = {}
        if boxes is not None:
            for classIndex in boxes.cls:
                className: str = classNames[int(classIndex)]
                classCountsDictionary[className] = classCountsDictionary.get(className, 0) + 1
        outputPartsList: list[str] = []
        for className, count in sorted(classCountsDictionary.items()):
            outputPartsList.append(f"{count} {className}s" if count > 1 else f"{count} {className}")
        processingTimeMilliseconds: float = 0.0
        if hasattr(result, 'speed') and result.speed:
            processingTimeMilliseconds = result.speed.get('inference', 0) + result.speed.get('postprocess', 0)
        outputLineString: str = (
            f"video 1/1 (frame {frameIndex + 1}/{totalFrames}) "
            f"{self.detectionPlan.videoSource}: "
            f"{result.orig_shape[1]}x{result.orig_shape[0]} "
            f"{', '.join(outputPartsList)}, "
            f"{processingTimeMilliseconds:.1f}ms"
        )
        self.detectionOutputSignal.emit(outputLineString + '\n')

    def Cancel(self) -> None:
        """取消检测任务"""
        self.isCancelled = True
        self.detectionSystem.CancelDetection()

    def GetResult(self) -> Result:
        """获取检测结果"""
        return self.detectionSystem.result


class DetectionWidget(QWidget):
    """检测标签页 - 显示实时视频检测画面"""
    detectionCompleted = Signal(object)

    def __init__(self, detectionSystem: VehicleDetectionSystem) -> None:
        super().__init__()
        self.detectionSystem = detectionSystem
        self.detectionPlan: DetectionPlan = detectionSystem.detectionPlan
        self.detectionWorker: Optional[DetectionWorker] = None
        self.result: Optional[Result] = None
        self.visualizationWidget: Optional[VisualizationWidget] = None

        mainLayout: QVBoxLayout = QVBoxLayout(self)
        buttonLayout: QHBoxLayout = QHBoxLayout()

        self.startButton: QPushButton = QPushButton("开始检测")
        self.startButton.clicked.connect(self.OnStartDetectionClicked)

        self.cancelButton: QPushButton = QPushButton("取消检测")
        self.cancelButton.clicked.connect(self.OnCancelDetectionClicked)
        self.cancelButton.setEnabled(False)

        buttonLayout.addWidget(self.startButton)
        buttonLayout.addWidget(self.cancelButton)
        buttonLayout.addStretch()

        self.progressBar: QProgressBar = QProgressBar()
        self.progressBar.setVisible(False)

        self.outputTextEdit: QTextEdit = QTextEdit()
        self.outputTextEdit.setReadOnly(True)
        self.outputTextEdit.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")

        mainLayout.addLayout(buttonLayout)
        mainLayout.addWidget(self.progressBar)
        mainLayout.addWidget(self.outputTextEdit)

    def SetVisualizationWidget(self, widget: VisualizationWidget) -> None:
        """设置可视化组件的引用"""
        self.visualizationWidget = widget

    def OnStartDetectionClicked(self) -> None:
        """处理开始检测按钮点击事件"""
        if self.detectionPlan.yoloModelPath is None or self.detectionPlan.videoSource is None:
            QMessageBox.warning(self, "警告", "请先在配置页面加载检测计划配置")
            return

        if self.detectionWorker is not None:
            self.detectionWorker.deleteLater()
            self.detectionWorker = None

        import gc
        gc.collect()

        self.outputTextEdit.clear()
        self.startButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        self.progressBar.setVisible(True)
        self.progressBar.setRange(0, 0)

        self.detectionWorker = DetectionWorker(self.detectionSystem)
        self.detectionWorker.detectionOutputSignal.connect(self.OnDetectionOutputReceived)
        self.detectionWorker.detectionFinishedSignal.connect(self.OnDetectionFinished)
        self.detectionWorker.start()

    def OnCancelDetectionClicked(self) -> None:
        """处理取消检测按钮点击事件"""
        if self.detectionWorker is not None:
            self.detectionWorker.Cancel()
            self.outputTextEdit.append("正在取消检测...\n")

    def OnDetectionOutputReceived(self, outputText: str) -> None:
        """处理检测输出接收"""
        self.outputTextEdit.append(outputText)
        scrollBar = self.outputTextEdit.verticalScrollBar()
        scrollBar.setValue(scrollBar.maximum())

    def OnDetectionFinished(self, success: bool, message: str) -> None:
        """处理检测完成事件"""
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        self.progressBar.setVisible(False)

        if success:
            self.result = self.detectionWorker.GetResult()
            performanceMetrics = self.result.performanceMetrics

            self.outputTextEdit.append("=" * 40)
            self.outputTextEdit.append("检测统计摘要")
            self.outputTextEdit.append(f"YOLO 检测到的总车辆数: {performanceMetrics.totalDetectedVehicles}")
            self.outputTextEdit.append(f"成功提取统计信息的车辆数: {performanceMetrics.successfullyExtractedVehicles}")
            if performanceMetrics.totalDetectedVehicles > 0:
                self.outputTextEdit.append(f"提取成功率: {performanceMetrics.extractionSuccessRate:.2f}%")
            else:
                self.outputTextEdit.append("提取成功率: N/A")
            self.outputTextEdit.append("车型分布:")
            for vehicleType, count in performanceMetrics.vehicleTypeDistribution.items():
                self.outputTextEdit.append(f"  {vehicleType}: {count}")
            self.outputTextEdit.append("颜色分布:")
            for color, count in performanceMetrics.vehicleColorDistribution.items():
                self.outputTextEdit.append(f"  {color}: {count}")
            self.outputTextEdit.append("车道分布:")
            for laneName, vehicleCount in performanceMetrics.vehicleLaneDistribution.items():
                self.outputTextEdit.append(f"  {laneName}: {vehicleCount} 辆车")

            self.outputTextEdit.append("=" * 40)
            self.outputTextEdit.append("实时性评估（实际采样帧率）")
            self.outputTextEdit.append(f"视频时长: {performanceMetrics.videoDuration:.2f} 秒")
            self.outputTextEdit.append(f"视频原始帧率: {self.result.frameRate:.2f} FPS")
            self.outputTextEdit.append(f"视频总帧数: {self.result.totalFrame}")
            self.outputTextEdit.append(f"实际检测帧数: {performanceMetrics.processedFrames}（抽帧后）")
            self.outputTextEdit.append(f"总处理耗时: {performanceMetrics.totalProcessingTime:.2f} 秒")
            self.outputTextEdit.append(f"实际采样帧率: {performanceMetrics.actualFps:.2f} FPS（总帧数/总耗时）")
            if performanceMetrics.actualFps >= 15.0:
                self.outputTextEdit.append("评分: 满分（≥15 FPS）")
            else:
                scoreLost = int(max(0, 15 - int(performanceMetrics.actualFps)))
                self.outputTextEdit.append(f"评分: 扣 {scoreLost} 分（每低1FPS扣1分）")

            self.outputTextEdit.append("=" * 40)
            self.outputTextEdit.append("速度统计")
            self.outputTextEdit.append(f"平均速度: {performanceMetrics.averageVelocity:.2f} m/s")
            self.outputTextEdit.append(f"最大速度: {performanceMetrics.maxVelocity:.2f} m/s")
            self.outputTextEdit.append(f"最小速度: {performanceMetrics.minVelocity:.2f} m/s")

            self.outputTextEdit.append("=" * 40)
            self.outputTextEdit.append("检测质量统计")
            self.outputTextEdit.append(f"总检测框数: {performanceMetrics.totalBoundingBoxes}")
            self.outputTextEdit.append(f"每帧平均检测框数: {performanceMetrics.averageBoundingBoxesPerFrame:.2f}")
            self.outputTextEdit.append(f"有检测结果的帧数: {performanceMetrics.framesWithDetections}")
            self.outputTextEdit.append(f"无检测结果的帧数: {performanceMetrics.framesWithoutDetections}")

            self.outputTextEdit.append("=" * 40)
            self.outputTextEdit.append("处理时间统计")
            self.outputTextEdit.append(f"最长单帧处理耗时: {performanceMetrics.maxProcessingTime:.4f} s")
            self.outputTextEdit.append(f"最短单帧处理耗时: {performanceMetrics.minProcessingTime:.4f} s")

            self.detectionCompleted.emit(self.result)
            if self.visualizationWidget is not None:
                self.visualizationWidget.ResultUpdate(self.result)
        else:
            self.outputTextEdit.append(f"\n{message}")

        self.detectionWorker = None
