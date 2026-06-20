import numpy as np
from typing import (Optional,
                    List,
                    Dict)
from Scripts.Lane import Lane
import json


class DetectionPlan:
    """检测计划类，管理检测配置参数"""
    def __init__(self) -> None:
        """初始化检测计划"""
        self.name: str = "未命名配置"
        self.yoloModelPath: Optional[str] = None
        self.videoSource: Optional[str] = None
        self.vehicles: Optional[Dict[str, Dict[str, float]]] = None
        self.pointsPerspective: Optional[np.ndarray] = None
        self.pointsParallel: Optional[np.ndarray] = None
        self.timeOccupancyWindows: int = 5
        self.spaceOccupancyWindows: int = 5
        self.flowWindows: int = 60
        self.queueLengthWindows: int = 1
        self.laneList: List[Lane] = []

    def ToDictionary(self) -> dict:
        """将检测计划转换为字典"""
        return {
            "name": self.name,
            "yoloModelPath": self.yoloModelPath,
            "videoSource": self.videoSource,
            "vehicles": self.vehicles,
            "pointsPerspective": (self.pointsPerspective.tolist()),
            "pointsParallel": (self.pointsParallel.tolist()),
            "timeOccupancyWindows": self.timeOccupancyWindows,
            "spaceOccupancyWindows": self.spaceOccupancyWindows,
            "flowWindows": self.flowWindows,
            "queueLengthWindows": self.queueLengthWindows,
            "laneList": [
                {
                    "sectionName": lane.sectionName,
                    "linePoints": lane.linePoints.tolist() if isinstance(lane.linePoints, np.ndarray) else [],
                    "edgePoints": lane.edgePoints.tolist() if isinstance(lane.edgePoints, np.ndarray) else [],
                    "vehicles": lane.vehicles,
                    "timeOccupancyWindows": lane.timeOccupancyWindows,
                    "spaceOccupancyWindows": lane.spaceOccupancyWindows,
                    "flowWindows": lane.flowWindows,
                    "queueLengthWindows": lane.queueLengthWindows,
                }
                for lane in self.laneList]}

    def FromDictionary(self, dictionary: dict) -> None:
        """从字典恢复检测计划"""
        self.name: str = dictionary.get("name", "未命名配置")
        self.yoloModelPath: str = dictionary["yoloModelPath"]
        self.videoSource: str = dictionary["videoSource"]
        self.vehicles: dict[str, dict[str, float]] = dictionary["vehicles"]
        self.pointsPerspective: np.ndarray = np.array(dictionary["pointsPerspective"], dtype=np.float32)
        self.pointsParallel: np.ndarray = np.array(dictionary["pointsParallel"], dtype=np.float32)
        self.timeOccupancyWindows: int = dictionary["timeOccupancyWindows"]
        self.spaceOccupancyWindows: int = dictionary["spaceOccupancyWindows"]
        self.flowWindows: int = dictionary["flowWindows"]
        self.queueLengthWindows: int = dictionary["queueLengthWindows"]
        self.laneList = []
        for laneDictionary in dictionary.get("laneList", []):
            laneObject: Lane = Lane(
                sectionName=laneDictionary["sectionName"],
                linePoints=np.array(laneDictionary["linePoints"], dtype=np.float32),
                edgePoints=np.array(laneDictionary["edgePoints"], dtype=np.float32),
                vehicles=laneDictionary["vehicles"],
                timeOccupancyWindows=laneDictionary["timeOccupancyWindows"],
                spaceOccupancyWindows=laneDictionary["spaceOccupancyWindows"],
                flowWindows=laneDictionary["flowWindows"],
                queueLengthWindows=laneDictionary["queueLengthWindows"],
            )
            self.laneList.append(laneObject)

    def ToJson(self, filePath: str) -> None:
        """将检测计划导出为JSON文件"""
        with open(filePath, "w", encoding="utf-8") as fileHandle:
            json.dump(self.ToDictionary(), fileHandle, ensure_ascii=False, indent=2)

    def FromJson(self, filePath: str) -> None:
        """从JSON文件导入检测计划"""
        with open(filePath, "r", encoding="utf-8") as fileHandle:
            loadedDictionary: dict = json.load(fileHandle)
        self.FromDictionary(loadedDictionary)
