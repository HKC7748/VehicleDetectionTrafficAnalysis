import cv2
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import (Results,
                                        Boxes)
from Scripts.DetectionPlan import DetectionPlan
from Scripts.Result import Result
from Scripts.VehicleInformation import VehicleDetectionInformation
from Scripts.Tool import Tool


class VehicleDetectionSystem:
    """检测系统"""
    def __init__(self):
        self.result: Result = Result()
        self.isRunning: bool = False
        self.detectionPlan: DetectionPlan = DetectionPlan()

    def StartDetection(self, progressCallback=None) -> None:
        """开始检测，只收集原始检测数据，不进行统计计算"""
        import gc
        gc.collect()
        self.isRunning = True
        self.result: Result = Result()

        self.result.videoSource = self.detectionPlan.videoSource
        self.result.vehicles = self.detectionPlan.vehicles
        self.result.pointsPerspective = self.detectionPlan.pointsPerspective
        self.result.pointsParallel = self.detectionPlan.pointsParallel
        self.result.laneList = self.detectionPlan.laneList

        yoloModel: YOLO = YOLO(self.detectionPlan.yoloModelPath)
        self.result.classNames = yoloModel.names

        capture: cv2.VideoCapture = cv2.VideoCapture(self.detectionPlan.videoSource)
        self.result.totalFrame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.result.frameRate = capture.get(cv2.CAP_PROP_FPS)
        for lane in self.result.laneList:
            lane.totalFrame = self.result.totalFrame
            lane.frameRate = self.result.frameRate

        detectInterval: int = 3
        detectedFrameCount: int = 0

        currentFrameIndex: int = 0
        while currentFrameIndex < self.result.totalFrame and self.isRunning:
            frameStartTime: float = time.time()

            ret, frame = capture.read()
            if not ret:
                break

            if currentFrameIndex % detectInterval == 0:
                results: list[Results] = yoloModel.track(
                    source=frame,
                    tracker="bytetrack.yaml",
                    show=False,
                    conf=0.25,
                    iou=0.5,
                    classes=[2, 3, 5, 7],
                    verbose=False,
                    persist=True
                )

                result: Results = results[0]
                boxes: Boxes = result.boxes
                frameImageArray: np.ndarray = result.orig_img

                if boxes is not None:
                    for boxIndex in range(len(boxes)):
                        if boxes.id is None or boxes.conf is None or boxes.cls is None or boxes.xywh is None:
                            continue

                        trackIdentifier: int = int(boxes.id[boxIndex])
                        classIdentifier: int = int(boxes.cls[boxIndex])
                        vehicleType: str = self.result.classNames[classIdentifier]
                        probability: float = float(boxes.conf[boxIndex])
                        xCenter: float = float(boxes.xywh[boxIndex][0])
                        yCenter: float = float(boxes.xywh[boxIndex][1])
                        width: float = float(boxes.xywh[boxIndex][2])
                        height: float = float(boxes.xywh[boxIndex][3])

                        vehicleColor: str = Tool.GetVehicleColor(
                            imageArray=frameImageArray,
                            x=xCenter,
                            y=yCenter,
                            w=width,
                            h=height
                        )

                        if trackIdentifier not in self.result.vehicleDetectionInformationDictionary:
                            self.result.vehicleDetectionInformationDictionary[trackIdentifier] = (
                                VehicleDetectionInformation(
                                    trackIdentifier=trackIdentifier,
                                    vehicleCategory=vehicleType,
                                    frameIndex=currentFrameIndex,
                                    color=vehicleColor,
                                    probability=probability,
                                    xCoordinate=xCenter,
                                    yCoordinate=yCenter,
                                    width=width,
                                    height=height
                                )
                            )
                        else:
                            self.result.vehicleDetectionInformationDictionary[trackIdentifier].AddFrameInformation(
                                frameIndex=currentFrameIndex,
                                color=vehicleColor,
                                probability=probability,
                                xCoordinate=xCenter,
                                yCoordinate=yCenter,
                                width=width,
                                height=height)

                frameEndTime: float = time.time()
                elapsedTime: float = frameEndTime - frameStartTime
                self.result.performanceMetrics.processingTimes.append(elapsedTime)

                if progressCallback is not None:
                    progressCallback(currentFrameIndex, self.result.totalFrame, result)

                detectedFrameCount += 1

            currentFrameIndex += 1
        capture.release()
        self.Finalize()

        self.isRunning = False

    def CancelDetection(self) -> None:
        """取消检测"""
        self.isRunning = False

    def Finalize(self) -> None:
        """执行所有统计信息计算，只在检测完成后运行一次"""
        performanceMetrics = self.result.performanceMetrics
        for vehicleDetectionInformation in self.result.vehicleDetectionInformationDictionary.values():
            if len(vehicleDetectionInformation.frameIndexList) < 10:
                continue
            vehicleStatisticsInformation = Tool.VehicleDetectionTransform(
                sourcePoints=self.result.pointsPerspective,
                destinationPoints=self.result.pointsParallel,
                vehicleDetectionInformation=vehicleDetectionInformation
            )
            vehicleStatisticsInformation.Update()
            self.result.vehicleStatisticsInformationDictionary[vehicleDetectionInformation.trackIdentifier] = vehicleStatisticsInformation

        for lane in self.result.laneList:
            for vehicleStatsInfo in self.result.vehicleStatisticsInformationDictionary.values():
                lane.AddVehicleStatisticsInformation(vehicleStatsInfo)
            lane.Update()
        performanceMetrics.totalDetectedVehicles = len(self.result.vehicleDetectionInformationDictionary)
        performanceMetrics.successfullyExtractedVehicles = len(self.result.vehicleStatisticsInformationDictionary)
        performanceMetrics.extractionSuccessRate = (performanceMetrics.successfullyExtractedVehicles / performanceMetrics.totalDetectedVehicles * 100 if performanceMetrics.totalDetectedVehicles > 0 else 0.0)
        performanceMetrics.processedFrames = len(performanceMetrics.processingTimes)
        performanceMetrics.videoDuration = self.result.totalFrame / self.result.frameRate if self.result.frameRate > 0 else 0.0

        if performanceMetrics.processingTimes:
            performanceMetrics.totalProcessingTime = sum(performanceMetrics.processingTimes)
            performanceMetrics.actualFps = self.result.totalFrame / performanceMetrics.totalProcessingTime
            performanceMetrics.maxProcessingTime = max(performanceMetrics.processingTimes)  # 补上
            performanceMetrics.minProcessingTime = min(performanceMetrics.processingTimes)  # 补上

        for vehicleStatsInfo in self.result.vehicleStatisticsInformationDictionary.values():
            vehicleCategory: str = vehicleStatsInfo.vehicleCategory
            performanceMetrics.vehicleTypeDistribution[vehicleCategory] = (performanceMetrics.vehicleTypeDistribution.get(vehicleCategory, 0) + 1)
            performanceMetrics.vehicleColorDistribution[vehicleStatsInfo.color] = (performanceMetrics.vehicleColorDistribution.get(vehicleStatsInfo.color, 0) + 1)
        for lane in self.result.laneList:
            laneName: str = lane.sectionName
            performanceMetrics.vehicleLaneDistribution[laneName] = len(lane.crossLineEventList)

        allVelocities: list[float] = []
        for vehicleStatsInfo in self.result.vehicleStatisticsInformationDictionary.values():
            allVelocities.extend(vehicleStatsInfo.estimatedVelocityList)
        if allVelocities:
            performanceMetrics.averageVelocity = sum(allVelocities) / len(allVelocities)
            performanceMetrics.maxVelocity = max(allVelocities)
            performanceMetrics.minVelocity = min(allVelocities)

        performanceMetrics.totalBoundingBoxes = sum(len(vehicleDetectionInfo.frameIndexList) for vehicleDetectionInfo in self.result.vehicleDetectionInformationDictionary.values())
        performanceMetrics.averageBoundingBoxesPerFrame = (performanceMetrics.totalBoundingBoxes / performanceMetrics.processedFrames if performanceMetrics.processedFrames > 0 else 0.0)
        performanceMetrics.framesWithDetections = len([t for t in performanceMetrics.processingTimes if t > 0])
        performanceMetrics.framesWithoutDetections = performanceMetrics.processedFrames - performanceMetrics.framesWithDetections
