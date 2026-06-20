import numpy as np
import math
import cv2
import pandas as pd
from Scripts.Tool import Tool
from Scripts.VehicleInformation import VehicleStatisticsInformation


class CrossLineEvent:
    """过线事件类，记录车辆过检测线的事件信息"""

    def __init__(self, trackIdentifier: int, frameIndex: int, vehicleCategory: str, color: str, velocity: float) -> None:
        self.trackIdentifier: int = trackIdentifier
        self.frameIndex: int = frameIndex
        self.vehicleCategory: str = vehicleCategory
        self.color: str = color
        self.velocity: float = velocity


class VehicleHeadWayInformation:
    """车头时距信息类，记录两车之间的车头时距"""

    def __init__(self, trackIdentifierFirst: int, trackIdentifierSecond: int, headFrameSpacing: int | None) -> None:
        self.trackIdentifierFirst: int = trackIdentifierFirst
        self.trackIdentifierSecond: int = trackIdentifierSecond
        self.headFrameSpacing: int | None = headFrameSpacing


class VehicleSpacingInformation:
    """车头间距信息类，记录两车之间的距离"""

    def __init__(self, trackIdentifierFirst: int, trackIdentifierSecond: int, spacingDistance: float | None) -> None:
        self.trackIdentifierFirst: int = trackIdentifierFirst
        self.trackIdentifierSecond: int = trackIdentifierSecond
        self.spacingDistance: float | None = spacingDistance


class Lane:
    """车道类，管理单个车道的检测和统计信息"""

    def __init__(self,
                 sectionName: str,
                 linePoints: np.ndarray,
                 edgePoints: np.ndarray,
                 vehicles: dict[str, dict[str, float]],
                 timeOccupancyWindows: int,
                 spaceOccupancyWindows: int,
                 flowWindows: int,
                 queueLengthWindows: int,
                 totalFrame: int = 150,
                 frameRate: int = 15) -> None:
        self.sectionName: str = sectionName
        self.linePoints: np.ndarray = linePoints
        self.edgePoints: np.ndarray = edgePoints
        self.vehicles: dict[str, dict[str, float]] = vehicles
        self.area: float = Tool.PolygonArea(edgePoints)
        self.vehicleStatisticsInformationDictionary: dict[int, VehicleStatisticsInformation] = {}
        self.crossLineEventList: list[CrossLineEvent] = []
        self.vehicleHeadWayList: list[VehicleHeadWayInformation] = []
        self.vehicleSpacingList: list[VehicleSpacingInformation] = []
        self.timeOccupancyList: list[tuple[int, int, float]] = []
        self.timeOccupancyWindows: int = timeOccupancyWindows
        self.spaceOccupancyList: list[tuple[int, int, float]] = []
        self.spaceOccupancyWindows: int = spaceOccupancyWindows
        self.flowList: list[tuple[int, int, int]] = []
        self.flowWindows: int = flowWindows
        self.queueLengthList: list[tuple[int, int, float | None]] = []
        self.queueLengthWindows: int = queueLengthWindows
        self.frameRate: int = frameRate
        self.totalFrame: int = totalFrame

    def Update(self) -> None:
        """更新车道统计数据"""
        if self.totalFrame == 0:
            return

        self.crossLineEventList.sort(key=lambda crossLineEvent: crossLineEvent.frameIndex)
        self.vehicleHeadWayList.clear()
        self.vehicleSpacingList.clear()
        for eventIndex in range(len(self.crossLineEventList)):
            if eventIndex == len(self.crossLineEventList) - 1:
                break

            currentTrackIdentifier: int = self.crossLineEventList[eventIndex].trackIdentifier
            nextTrackIdentifier: int = self.crossLineEventList[eventIndex + 1].trackIdentifier
            frameDifference: int = self.crossLineEventList[eventIndex + 1].frameIndex - self.crossLineEventList[eventIndex].frameIndex
            self.vehicleHeadWayList.append(VehicleHeadWayInformation(
                trackIdentifierFirst=currentTrackIdentifier,
                trackIdentifierSecond=nextTrackIdentifier,
                headFrameSpacing=frameDifference
            ))

            currentVehicleStatisticsInformation: VehicleStatisticsInformation = self.vehicleStatisticsInformationDictionary[currentTrackIdentifier]
            nextVehicleStatisticsInformation: VehicleStatisticsInformation = self.vehicleStatisticsInformationDictionary[nextTrackIdentifier]
            endFrameIndex: int = self.crossLineEventList[eventIndex].frameIndex
            if nextVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex) is None:
                firstFrameIndex: int = nextVehicleStatisticsInformation.fittedFrameIndexList[0]
                differenceX: float = nextVehicleStatisticsInformation.GetPositionInFrame(firstFrameIndex)[0] - currentVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[0]
                differenceY: float = nextVehicleStatisticsInformation.GetPositionInFrame(firstFrameIndex)[1] - currentVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[1]
            else:
                differenceX: float = nextVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[0] - currentVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[0]
                differenceY: float = nextVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[1] - currentVehicleStatisticsInformation.GetPositionInFrame(endFrameIndex)[1]
            distance: float = math.sqrt(differenceX ** 2 + differenceY ** 2)
            self.vehicleSpacingList.append(VehicleSpacingInformation(
                trackIdentifierFirst=currentTrackIdentifier,
                trackIdentifierSecond=nextTrackIdentifier,
                spacingDistance=distance
            ))

        self.timeOccupancyList.clear()
        allWindowsCount: int = self.totalFrame
        windowSize: int = int(self.timeOccupancyWindows * self.frameRate)
        timeOccupancyIndexCount: int = allWindowsCount // windowSize
        for windowIndex in range(timeOccupancyIndexCount):
            startFrameIndex: int = windowIndex * windowSize
            endFrameIndex: int = startFrameIndex + windowSize
            occupiedFrameCount: int = 0
            for frameIndex in range(startFrameIndex, endFrameIndex):
                isOccupied: bool = False
                for vehicleStatisticsInformation in self.vehicleStatisticsInformationDictionary.values():
                    position = vehicleStatisticsInformation.GetPositionInFrame(frameIndex)
                    if position is None:
                        continue
                    if cv2.pointPolygonTest(self.edgePoints.astype(np.int32), position, False) >= 0:
                        isOccupied = True
                        break
                if isOccupied:
                    occupiedFrameCount += 1
            timeOccupancyValue: float = occupiedFrameCount / windowSize * 100
            self.timeOccupancyList.append((startFrameIndex, endFrameIndex, timeOccupancyValue))

        self.spaceOccupancyList.clear()
        allWindowsCount = self.totalFrame
        windowSize = int(self.spaceOccupancyWindows * self.frameRate)
        spaceOccupancyIndexCount: int = allWindowsCount // windowSize
        for windowIndex in range(spaceOccupancyIndexCount):
            startFrameIndex = windowIndex * windowSize
            endFrameIndex = startFrameIndex + windowSize
            totalAreaSum: float = 0.0
            for frameIndex in range(startFrameIndex, endFrameIndex):
                frameTotalArea: float = 0.0
                for vehicleStatisticsInformation in self.vehicleStatisticsInformationDictionary.values():
                    position = vehicleStatisticsInformation.GetPositionInFrame(frameIndex)
                    if position is None:
                        continue
                    if cv2.pointPolygonTest(self.edgePoints.astype(np.int32), position, False) < 0:
                        continue
                    vehicleLength: float = self.vehicles[vehicleStatisticsInformation.vehicleCategory]["length"]
                    vehicleWidth: float = self.vehicles[vehicleStatisticsInformation.vehicleCategory]["width"]
                    frameTotalArea += vehicleLength * vehicleWidth
                totalAreaSum += frameTotalArea
            averageSpaceOccupancy: float = totalAreaSum / windowSize / self.area * 100
            self.spaceOccupancyList.append((startFrameIndex, endFrameIndex, averageSpaceOccupancy))

        self.flowList.clear()
        allWindowsCount = self.totalFrame
        windowSize = int(self.flowWindows * self.frameRate)
        flowIndexCount: int = allWindowsCount // windowSize
        for windowIndex in range(flowIndexCount):
            startFrameIndex = windowIndex * windowSize
            endFrameIndex = startFrameIndex + windowSize
            flowCount: int = 0
            for crossLineEventObject in self.crossLineEventList:
                if startFrameIndex <= crossLineEventObject.frameIndex < endFrameIndex:
                    flowCount += 1
            self.flowList.append((startFrameIndex, endFrameIndex, flowCount))

        self.queueLengthList.clear()
        allWindowsCount = self.totalFrame
        windowSize = int(self.queueLengthWindows * self.frameRate)
        queueLengthIndexCount: int = allWindowsCount // windowSize
        for windowIndex in range(queueLengthIndexCount):
            startFrameIndex = windowIndex * windowSize
            endFrameIndex = startFrameIndex + windowSize
            totalQueueLengthSum: float = 0.0
            validFrameCount: int = 0
            for frameIndex in range(startFrameIndex, endFrameIndex):
                frameQueueCandidates: list[float] = []
                for vehicleStatisticsInformation in self.vehicleStatisticsInformationDictionary.values():
                    position = vehicleStatisticsInformation.GetPositionInFrame(frameIndex)
                    if position is None:
                        continue
                    crossedEvents = [eventObject for eventObject in self.crossLineEventList if eventObject.trackIdentifier == vehicleStatisticsInformation.trackIdentifier and eventObject.frameIndex <= frameIndex]
                    if crossedEvents:
                        continue
                    if frameIndex in vehicleStatisticsInformation.fittedFrameIndexList:
                        velocityIndex = vehicleStatisticsInformation.fittedFrameIndexList.index(frameIndex)
                        if vehicleStatisticsInformation.estimatedVelocityList[velocityIndex] < 0.5:
                            distanceValue: float = math.sqrt((position[0] - self.linePoints[0][0]) ** 2 + (position[1] - self.linePoints[0][1]) ** 2)
                            frameQueueCandidates.append(distanceValue)
                if frameQueueCandidates:
                    totalQueueLengthSum += max(frameQueueCandidates)
                    validFrameCount += 1
            averageQueueLength: float = totalQueueLengthSum / validFrameCount if validFrameCount > 0 else 0.0
            self.queueLengthList.append((startFrameIndex, endFrameIndex, averageQueueLength))

    def AddVehicleStatisticsInformation(self, vehicleStatisticsInformation: VehicleStatisticsInformation) -> None:
        """添加车辆统计信息"""
        for pointIndex in range(len(vehicleStatisticsInformation.fittedFrameIndexList) - 1):
            pointP: tuple[float, float] = (vehicleStatisticsInformation.fittedCoordinateXList[pointIndex], vehicleStatisticsInformation.fittedCoordinateYList[pointIndex])
            pointQ: tuple[float, float] = (vehicleStatisticsInformation.fittedCoordinateXList[pointIndex + 1], vehicleStatisticsInformation.fittedCoordinateYList[pointIndex + 1])
            pointA: tuple[float, float] = (self.linePoints[0, 0], self.linePoints[0, 1])
            pointB: tuple[float, float] = (self.linePoints[1, 0], self.linePoints[1, 1])
            if Tool.SegmentIntersect(pointA=pointA, pointB=pointB, pointP=pointP, pointQ=pointQ):
                self.crossLineEventList.append(CrossLineEvent(
                    trackIdentifier=vehicleStatisticsInformation.trackIdentifier,
                    frameIndex=vehicleStatisticsInformation.fittedFrameIndexList[pointIndex],
                    vehicleCategory=vehicleStatisticsInformation.vehicleCategory,
                    color=vehicleStatisticsInformation.color,
                    velocity=vehicleStatisticsInformation.estimatedVelocityList[pointIndex]
                ))
                self.vehicleStatisticsInformationDictionary[vehicleStatisticsInformation.trackIdentifier] = vehicleStatisticsInformation

    def GetTimeOccupancyDataFrame(self) -> pd.DataFrame:
        """导出时间占有率"""
        if not self.timeOccupancyList:
            return pd.DataFrame(columns=["起始时间(s)", "结束时间(s)", "时间占有率(%)"])
        statisticsData: list[dict] = []
        for occupancyEntry in self.timeOccupancyList:
            startFrameIndex: int = occupancyEntry[0]
            endFrameIndex: int = occupancyEntry[1]
            timeOccupancyValue: float = occupancyEntry[2]
            statisticsData.append({
                "起始时间(s)": round(startFrameIndex / self.frameRate, 3),
                "结束时间(s)": round(endFrameIndex / self.frameRate, 3),
                "时间占有率(%)": round(timeOccupancyValue, 3)
            })
        statisticsDataFrame: pd.DataFrame = pd.DataFrame(statisticsData)
        return statisticsDataFrame

    def GetSpaceOccupancyDataFrame(self) -> pd.DataFrame:
        """导出空间占有率"""
        if not self.spaceOccupancyList:
            return pd.DataFrame(columns=["起始时间(s)", "结束时间(s)", "空间占有率(%)"])
        statisticsData: list[dict] = []
        for occupancyEntry in self.spaceOccupancyList:
            startFrameIndex: int = occupancyEntry[0]
            endFrameIndex: int = occupancyEntry[1]
            spaceOccupancyValue: float = occupancyEntry[2]
            statisticsData.append({
                "起始时间(s)": round(startFrameIndex / self.frameRate, 3),
                "结束时间(s)": round(endFrameIndex / self.frameRate, 3),
                "空间占有率(%)": round(spaceOccupancyValue, 3)
            })
        statisticsDataFrame: pd.DataFrame = pd.DataFrame(statisticsData)
        return statisticsDataFrame

    def GetFlowDataFrame(self) -> pd.DataFrame:
        """导出流量"""
        if not self.flowList:
            return pd.DataFrame(columns=["起始时间(s)", "结束时间(s)", "流量(辆/分钟)"])
        statisticsData: list[dict] = []
        for flowEntry in self.flowList:
            startFrameIndex: int = flowEntry[0]
            endFrameIndex: int = flowEntry[1]
            flowCount: int = flowEntry[2]
            statisticsData.append({
                "起始时间(s)": round(startFrameIndex / self.frameRate, 3),
                "结束时间(s)": round(endFrameIndex / self.frameRate, 3),
                "流量(辆/分钟)": flowCount
            })
        statisticsDataFrame: pd.DataFrame = pd.DataFrame(statisticsData)
        return statisticsDataFrame

    def GetQueueLengthDataFrame(self) -> pd.DataFrame:
        """导出排队长度"""
        if not self.queueLengthList:
            return pd.DataFrame(columns=["起始时间(s)", "结束时间(s)", "排队长度(m)"])
        statisticsData: list[dict] = []
        for queueEntry in self.queueLengthList:
            startFrameIndex: int = queueEntry[0]
            endFrameIndex: int = queueEntry[1]
            queueLengthValue: float | None = queueEntry[2]
            statisticsData.append({
                "起始时间(s)": round(startFrameIndex / self.frameRate, 3),
                "结束时间(s)": round(endFrameIndex / self.frameRate, 3),
                "排队长度(m)": round(queueLengthValue, 3) if queueLengthValue is not None else 0.0
            })
        statisticsDataFrame: pd.DataFrame = pd.DataFrame(statisticsData)
        return statisticsDataFrame

    def GetSectionPassingDataFrame(self) -> pd.DataFrame:
        """导出断面过车数据（车辆穿过检测线时）"""
        if not self.crossLineEventList:
            return pd.DataFrame(columns=[
                "车辆ID", "时间(s)", "车辆类型", "车身颜色",
                "到达/离去", "速度(m/s)", "车头时距(s)", "间距(m)"
            ])
        sectionPassingData: list[dict] = []
        for crossLineEventObject in self.crossLineEventList:
            trackIdentifier: int = crossLineEventObject.trackIdentifier
            headWay: float = 0.0
            spacing: float | None = None
            timeInSeconds: float = crossLineEventObject.frameIndex / self.frameRate
            for headWayInformation in self.vehicleHeadWayList:
                if headWayInformation.trackIdentifierFirst == trackIdentifier:
                    headWay = float(headWayInformation.headFrameSpacing) / self.frameRate
                    break
            for spacingInformation in self.vehicleSpacingList:
                if spacingInformation.trackIdentifierFirst == trackIdentifier:
                    spacing = float(spacingInformation.spacingDistance) / self.frameRate
                    break
            sectionPassingData.append({
                "车辆ID": trackIdentifier,
                "时间(s)": round(timeInSeconds, 3),
                "车辆类型": crossLineEventObject.vehicleCategory,
                "车身颜色": crossLineEventObject.color,
                "到达/离去": "到达",
                "速度(m/s)": round(crossLineEventObject.velocity, 3),
                "车头时距(s)": round(headWay, 3),
                "间距(m)": round(spacing, 3) if spacing is not None else 0.0
            })
        sectionPassingDataFrame: pd.DataFrame = pd.DataFrame(sectionPassingData)
        return sectionPassingDataFrame

    def ExportAllDataToCSV(self, exportDirectoryPath: str) -> None:
        """导出所有统计数据到CSV文件"""
        import os
        passingDataDirectory: str = os.path.join(exportDirectoryPath, "5类断面过车数据")
        statisticsDataDirectory: str = os.path.join(exportDirectoryPath, "3类统计数据")
        safeFileName: str = self.sectionName.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")
        laneSectionPassingDataDirectory: str = os.path.join(passingDataDirectory, f"{safeFileName}数据")
        laneSectionStatisticsDataDirectory: str = os.path.join(statisticsDataDirectory, f"{safeFileName}数据")
        os.makedirs(laneSectionPassingDataDirectory, exist_ok=True)
        os.makedirs(laneSectionStatisticsDataDirectory, exist_ok=True)
        sectionPassingDataFrame: pd.DataFrame = self.GetSectionPassingDataFrame()
        sectionPassingDataFrame.to_csv(
            os.path.join(laneSectionPassingDataDirectory, f"{safeFileName}_断面过车数据.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        timeOccupancyDataFrame: pd.DataFrame = self.GetTimeOccupancyDataFrame()
        timeOccupancyDataFrame.to_csv(
            os.path.join(laneSectionStatisticsDataDirectory, f"{safeFileName}_时间占有率.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        spaceOccupancyDataFrame: pd.DataFrame = self.GetSpaceOccupancyDataFrame()
        spaceOccupancyDataFrame.to_csv(
            os.path.join(laneSectionStatisticsDataDirectory, f"{safeFileName}_空间占有率.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        flowDataFrame: pd.DataFrame = self.GetFlowDataFrame()
        flowDataFrame.to_csv(
            os.path.join(laneSectionStatisticsDataDirectory, f"{safeFileName}_流量.csv"),
            index=False,
            encoding="utf-8-sig"
        )
        queueLengthDataFrame: pd.DataFrame = self.GetQueueLengthDataFrame()
        queueLengthDataFrame.to_csv(
            os.path.join(laneSectionStatisticsDataDirectory, f"{safeFileName}_排队长度.csv"),
            index=False,
            encoding="utf-8-sig"
        )
