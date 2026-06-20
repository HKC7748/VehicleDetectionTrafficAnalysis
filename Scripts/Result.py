import numpy as np
import cv2
from Scripts.VehicleInformation import (VehicleDetectionInformation,
                                        VehicleStatisticsInformation)
from Scripts.Lane import Lane
import pandas as pd


class VehicleDetectionInformationInFrame:
    """车辆检测信息帧内类，存储单帧内的车辆检测信息"""

    def __init__(self,
                 trackIdentifierList: list[int],
                 vehicleTypeList: list[str],
                 vehicleColorList: list[str],
                 probabilityList: list[float],
                 xCoordinateList: list[float],
                 yCoordinateList: list[float],
                 widthList: list[float],
                 heightList: list[float]) -> None:
        self.trackIdentifierList: list[int] = trackIdentifierList
        self.vehicleTypeList: list[str] = vehicleTypeList
        self.vehicleColorList: list[str] = vehicleColorList
        self.probabilityList: list[float] = probabilityList
        self.xCoordinateList: list[float] = xCoordinateList
        self.yCoordinateList: list[float] = yCoordinateList
        self.widthList: list[float] = widthList
        self.heightList: list[float] = heightList


class VehicleStatisticsInformationInFrame:
    """车辆统计信息帧内类，存储单帧内的车辆统计信息"""

    def __init__(self,
                 trackIdentifierList: list[int],
                 vehicleNameList: list[str],
                 vehicleColorList: list[str],
                 xCoordinateList: list[float],
                 yCoordinateList: list[float],
                 vehicleStatisticsInformationDictionary: dict[int, VehicleStatisticsInformation] = None) -> None:
        self.trackIdentifierList: list[int] = trackIdentifierList
        self.vehicleNameList: list[str] = vehicleNameList
        self.vehicleColorList: list[str] = vehicleColorList
        self.xCoordinateList: list[float] = xCoordinateList
        self.yCoordinateList: list[float] = yCoordinateList
        self.vehicleStatisticsInformationDictionary: dict[int, VehicleStatisticsInformation] = vehicleStatisticsInformationDictionary or {}


class PerformanceMetrics:
    """性能指标统计类，存储非10类数据以外的所有评估指标"""

    def __init__(self) -> None:
        self.totalDetectedVehicles: int = 0
        self.successfullyExtractedVehicles: int = 0
        self.extractionSuccessRate: float = 0.0
        self.processedFrames: int = 0
        self.videoDuration: float = 0.0
        self.processingTimes: list[float] = []
        self.totalProcessingTime: float = 0.0
        self.actualFps: float = 0.0
        self.maxProcessingTime: float = 0.0
        self.minProcessingTime: float = 0.0
        self.vehicleTypeDistribution: dict[str, int] = {}
        self.vehicleColorDistribution: dict[str, int] = {}
        self.vehicleLaneDistribution: dict[str, int] = {}
        self.averageVelocity: float = 0.0
        self.maxVelocity: float = 0.0
        self.minVelocity: float = 0.0
        self.totalBoundingBoxes: int = 0
        self.averageBoundingBoxesPerFrame: float = 0.0
        self.framesWithDetections: int = 0
        self.framesWithoutDetections: int = 0


class Result:
    """结果类，管理检测结果和相关数据"""

    def __init__(self) -> None:
        self.videoSource: str = ""
        self.vehicles: dict[str, dict[str, float]] = {}
        self.pointsPerspective: np.ndarray = np.array([])
        self.pointsParallel: np.ndarray = np.array([])
        self.classNames: dict[int, str] = {}
        self.frameRate: int = 0
        self.totalFrame: int = 0
        self.vehicleDetectionInformationDictionary: dict[int, VehicleDetectionInformation] = {}
        self.vehicleStatisticsInformationDictionary: dict[int, VehicleStatisticsInformation] = {}
        self.laneList: list[Lane] = []
        self.performanceMetrics: PerformanceMetrics = PerformanceMetrics()

    def GetCaptureInFrame(self, frameIndex: int) -> np.ndarray:
        """读取视频帧"""
        captureObject: cv2.VideoCapture = cv2.VideoCapture(self.videoSource)
        captureObject.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        ret: bool
        frame: np.ndarray
        ret, frame = captureObject.read()
        captureObject.release()
        if not ret:
            raise RuntimeError(f"无法读取帧 {frameIndex}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def GetVehicleDetectionInformationInFrame(self, frameIndex: int) -> VehicleDetectionInformationInFrame:
        """获取帧内的车辆检测信息"""
        trackIdentifierList: list[int] = []
        vehicleTypeList: list[str] = []
        vehicleColorList: list[str] = []
        probabilityList: list[float] = []
        xCoordinateList: list[float] = []
        yCoordinateList: list[float] = []
        widthList: list[float] = []
        heightList: list[float] = []
        for vehicleDetectionInformation in self.vehicleDetectionInformationDictionary.values():
            if frameIndex in vehicleDetectionInformation.frameIndexList:
                indexPosition: int = vehicleDetectionInformation.frameIndexList.index(frameIndex)
                trackIdentifierList.append(vehicleDetectionInformation.trackIdentifier)
                vehicleTypeList.append(vehicleDetectionInformation.vehicleCategory)
                vehicleColorList.append(vehicleDetectionInformation.colorList[indexPosition])
                probabilityList.append(vehicleDetectionInformation.probabilityList[indexPosition])
                xCoordinateList.append(vehicleDetectionInformation.xCoordinateList[indexPosition])
                yCoordinateList.append(vehicleDetectionInformation.yCoordinateList[indexPosition])
                widthList.append(vehicleDetectionInformation.widthList[indexPosition])
                heightList.append(vehicleDetectionInformation.heightList[indexPosition])
        return VehicleDetectionInformationInFrame(
            trackIdentifierList=trackIdentifierList,
            vehicleTypeList=vehicleTypeList,
            vehicleColorList=vehicleColorList,
            probabilityList=probabilityList,
            xCoordinateList=xCoordinateList,
            yCoordinateList=yCoordinateList,
            widthList=widthList,
            heightList=heightList
        )

    def GetVehicleStatisticsInformationInFrame(self, frameIndex: int) -> VehicleStatisticsInformationInFrame:
        """获取帧内的车辆统计信息"""
        trackIdentifierList: list[int] = []
        vehicleNameList: list[str] = []
        vehicleColorList: list[str] = []
        xCoordinateList: list[float] = []
        yCoordinateList: list[float] = []
        vehicleStatisticsInformationDictionary: dict[int, VehicleStatisticsInformation] = {}
        for vehicleStatisticsInformation in self.vehicleStatisticsInformationDictionary.values():
            if frameIndex in vehicleStatisticsInformation.fittedFrameIndexList:
                indexPosition: int = vehicleStatisticsInformation.fittedFrameIndexList.index(frameIndex)
                trackIdentifierList.append(vehicleStatisticsInformation.trackIdentifier)
                vehicleNameList.append(vehicleStatisticsInformation.vehicleCategory)
                vehicleColorList.append(vehicleStatisticsInformation.color)
                xCoordinateList.append(vehicleStatisticsInformation.fittedCoordinateXList[indexPosition])
                yCoordinateList.append(vehicleStatisticsInformation.fittedCoordinateYList[indexPosition])
                vehicleStatisticsInformationDictionary[vehicleStatisticsInformation.trackIdentifier] = vehicleStatisticsInformation
        return VehicleStatisticsInformationInFrame(
            trackIdentifierList=trackIdentifierList,
            vehicleNameList=vehicleNameList,
            vehicleColorList=vehicleColorList,
            xCoordinateList=xCoordinateList,
            yCoordinateList=yCoordinateList,
            vehicleStatisticsInformationDictionary=vehicleStatisticsInformationDictionary
        )

    def GetDetectionFrameImage(self, frameIndex: int, targetWidth: int, targetHeight: int) -> np.ndarray:
        """获取检测帧图像"""
        frameImageArray: np.ndarray = self.GetCaptureInFrame(frameIndex)
        vehicleDetectionInformationInFrame: VehicleDetectionInformationInFrame = self.GetVehicleDetectionInformationInFrame(frameIndex)
        for informationIndex in range(len(vehicleDetectionInformationInFrame.trackIdentifierList)):
            trackIdentifier: int = vehicleDetectionInformationInFrame.trackIdentifierList[informationIndex]
            vehicleType: str = vehicleDetectionInformationInFrame.vehicleTypeList[informationIndex]
            vehicleColor: str = vehicleDetectionInformationInFrame.vehicleColorList[informationIndex]
            probability: float = vehicleDetectionInformationInFrame.probabilityList[informationIndex]
            xCenter: float = vehicleDetectionInformationInFrame.xCoordinateList[informationIndex]
            yCenter: float = vehicleDetectionInformationInFrame.yCoordinateList[informationIndex]
            width: float = vehicleDetectionInformationInFrame.widthList[informationIndex]
            height: float = vehicleDetectionInformationInFrame.heightList[informationIndex]
            x1Coordinate: int = int(xCenter - width / 2)
            y1Coordinate: int = int(yCenter - height / 2)
            x2Coordinate: int = int(xCenter + width / 2)
            y2Coordinate: int = int(yCenter + height / 2)
            cv2.rectangle(frameImageArray, (x1Coordinate, y1Coordinate), (x2Coordinate, y2Coordinate), (0, 255, 0), 2)
            labelText: str = f"{vehicleColor}{vehicleType} {trackIdentifier} {probability:.2f}"
            cv2.putText(frameImageArray, labelText, (x1Coordinate, y1Coordinate - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        resizedFrame: np.ndarray = cv2.resize(frameImageArray, (targetWidth, targetHeight))
        return resizedFrame

    def GetStatisticsFrameImage(self, frameIndex: int, targetWidth: int, targetHeight: int) -> np.ndarray:
        """获取统计帧图像"""
        mapImage: np.ndarray = cv2.imread("./Resource/map.png")
        if mapImage is None:
            mapImage = np.ones((970, 1450, 3), dtype=np.uint8) * 240
        mapOriginalHeight: int = mapImage.shape[0]
        vehicleStatisticsInformationInFrame: VehicleStatisticsInformationInFrame = self.GetVehicleStatisticsInformationInFrame(frameIndex)
        for trackIdentifier, vehicleStatisticsInformation in vehicleStatisticsInformationInFrame.vehicleStatisticsInformationDictionary.items():
            trajectoryPoints: list[list[int]] = []
            for trajectoryIndex in range(len(vehicleStatisticsInformation.fittedCoordinateXList)):
                trajectoryX: float = vehicleStatisticsInformation.fittedCoordinateXList[trajectoryIndex] * 10
                trajectoryY: float = mapOriginalHeight - vehicleStatisticsInformation.fittedCoordinateYList[trajectoryIndex] * 10
                trajectoryPoints.append([int(trajectoryX), int(trajectoryY)])
            if len(trajectoryPoints) > 1:
                trajectoryArray: np.ndarray = np.array(trajectoryPoints, dtype=np.int32)
                cv2.polylines(mapImage, [trajectoryArray], False, (0, 255, 0), 2)
        for informationIndex in range(len(vehicleStatisticsInformationInFrame.trackIdentifierList)):
            trackIdentifier: int = vehicleStatisticsInformationInFrame.trackIdentifierList[informationIndex]
            xCoordinate: float = vehicleStatisticsInformationInFrame.xCoordinateList[informationIndex] * 10
            yCoordinate: float = mapOriginalHeight - vehicleStatisticsInformationInFrame.yCoordinateList[informationIndex] * 10
            cv2.circle(mapImage, (int(xCoordinate), int(yCoordinate)), 15, (0, 0, 255), -1)
            labelText: str = f"{trackIdentifier}"
            cv2.putText(mapImage, labelText, (int(xCoordinate), int(yCoordinate)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        resizedMapImage: np.ndarray = cv2.resize(mapImage, (targetWidth, targetHeight))
        return resizedMapImage

    def GetVehicleTypeDistributionDataFrame(self) -> pd.DataFrame:
        """导出车辆类型分布数据"""
        if not self.performanceMetrics.vehicleTypeDistribution:
            return pd.DataFrame(columns=["车辆类型", "车辆数目"])
        return pd.DataFrame(
            list(self.performanceMetrics.vehicleTypeDistribution.items()),
            columns=["车辆类型", "车辆数目"]
        )

    def GetVehicleColorDistributionDataFrame(self) -> pd.DataFrame:
        """导出车辆颜色分布数据"""
        if not self.performanceMetrics.vehicleColorDistribution:
            return pd.DataFrame(columns=["车辆颜色", "车辆数目"])
        return pd.DataFrame(
            list(self.performanceMetrics.vehicleColorDistribution.items()),
            columns=["车辆颜色", "车辆数目"]
        )

    def GetVehicleLaneDistributionDataFrame(self) -> pd.DataFrame:
        """导出车辆所在车道分布数据"""
        if not self.performanceMetrics.vehicleLaneDistribution:
            return pd.DataFrame(columns=["车辆车道", "车辆数目"])
        return pd.DataFrame(
            list(self.performanceMetrics.vehicleLaneDistribution.items()),
            columns=["车辆车道", "车辆数目"]
        )

    def GetTrajectoryDataFrame(self) -> pd.DataFrame:
        """导出车辆轨迹数据（每车每秒一条）"""
        trajectoryData: list[dict] = []
        for vehicleStatisticsInformation in self.vehicleStatisticsInformationDictionary.values():
            sampleInterval: int = max(1, int(self.frameRate))
            for indexPosition in range(len(vehicleStatisticsInformation.fittedFrameIndexList)):
                frameIndex: int = vehicleStatisticsInformation.fittedFrameIndexList[indexPosition]
                if indexPosition % sampleInterval == 0:
                    trajectoryData.append({
                        "车辆ID": vehicleStatisticsInformation.trackIdentifier,
                        "时间(s)": round(frameIndex / self.frameRate, 3),
                        "坐标X(m)": round(vehicleStatisticsInformation.fittedCoordinateXList[indexPosition], 3),
                        "坐标Y(m)": round(vehicleStatisticsInformation.fittedCoordinateYList[indexPosition], 3),
                        "车牌": "未知"
                    })
        trajectoryDataFrame: pd.DataFrame = pd.DataFrame(trajectoryData)
        return trajectoryDataFrame

    def ExportTrajectoryData(self, exportDirectoryPath: str) -> None:
        """导出车辆轨迹数据到CSV文件"""
        import os
        trajectoryDataDirectory: str = os.path.join(exportDirectoryPath, "2类车辆轨迹数据")
        os.makedirs(trajectoryDataDirectory, exist_ok=True)
        trajectoryDataFrame: pd.DataFrame = self.GetTrajectoryDataFrame()
        trajectoryDataFrame.to_csv(
            os.path.join(trajectoryDataDirectory, "车辆轨迹数据.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    def GetPerformanceMetricsDataFrame(self) -> pd.DataFrame:
        """获取性能指标DataFrame"""
        performanceMetricsObject: PerformanceMetrics = self.performanceMetrics
        performanceData: list[dict] = [{
            "检测到的总车辆数": performanceMetricsObject.totalDetectedVehicles,
            "成功提取统计信息的车辆数": performanceMetricsObject.successfullyExtractedVehicles,
            "提取成功率(%)": round(performanceMetricsObject.extractionSuccessRate, 2),
            "实际处理帧数": performanceMetricsObject.processedFrames,
            "视频时长(s)": round(performanceMetricsObject.videoDuration, 2),
            "总处理耗时(s)": round(performanceMetricsObject.totalProcessingTime, 2),
            "实际采样帧率(FPS)": round(performanceMetricsObject.actualFps, 2),
            "最长单帧处理耗时(s)": round(performanceMetricsObject.maxProcessingTime, 4),
            "最短单帧处理耗时(s)": round(performanceMetricsObject.minProcessingTime, 4),
            "车型分布": str(performanceMetricsObject.vehicleTypeDistribution),
            "颜色分布": str(performanceMetricsObject.vehicleColorDistribution),
            "各车道车辆数": str(performanceMetricsObject.vehicleLaneDistribution),
            "平均速度(m/s)": round(performanceMetricsObject.averageVelocity, 2),
            "最大速度(m/s)": round(performanceMetricsObject.maxVelocity, 2),
            "最小速度(m/s)": round(performanceMetricsObject.minVelocity, 2),
            "总检测框数": performanceMetricsObject.totalBoundingBoxes,
            "每帧平均检测框数": round(performanceMetricsObject.averageBoundingBoxesPerFrame, 2),
            "有检测结果的帧数": performanceMetricsObject.framesWithDetections,
            "无检测结果的帧数": performanceMetricsObject.framesWithoutDetections
        }]
        performanceDataFrame: pd.DataFrame = pd.DataFrame(performanceData)
        return performanceDataFrame

    def ExportPerformanceMetricsData(self, exportDirectoryPath: str) -> None:
        """导出性能指标数据到CSV文件"""
        import os
        performanceDataDirectory: str = os.path.join(exportDirectoryPath, "性能指标数据")
        os.makedirs(performanceDataDirectory, exist_ok=True)
        performanceDataFrame: pd.DataFrame = self.GetPerformanceMetricsDataFrame()
        performanceDataFrame.to_csv(
            os.path.join(performanceDataDirectory, "性能指标.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    def ExportAllDataToDirectory(self, exportDirectoryPath: str) -> None:
        """导出所有数据到指定目录"""
        self.ExportPerformanceMetricsData(exportDirectoryPath)
        self.ExportTrajectoryData(exportDirectoryPath)
        for laneObject in self.laneList:
            laneObject.ExportAllDataToCSV(exportDirectoryPath)
