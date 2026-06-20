import cv2
import numpy as np
from Scripts.VehicleInformation import (VehicleDetectionInformation,
                                        VehicleStatisticsInformation)


class Tool:
    """工具类，提供各种静态辅助方法"""

    @staticmethod
    def PerspectiveToParallelTransform(sourcePoints: np.ndarray,
                                       destinationPoints: np.ndarray,
                                       points: np.ndarray) -> np.ndarray:
        """透视变换到平行变换"""
        sourcePointsArray: np.ndarray = sourcePoints.astype(np.float64)
        destinationPointsArray: np.ndarray = destinationPoints.astype(np.float64)
        pointsArray: np.ndarray = points.astype(np.float64)
        homographyMatrix, mask = cv2.findHomography(sourcePointsArray, destinationPointsArray, cv2.RANSAC, 3.0)
        if homographyMatrix is None:
            raise ValueError("无法计算有效的单应性矩阵")
        pointsParallelCalculated: np.ndarray = cv2.perspectiveTransform(
            pointsArray.reshape(-1, 1, 2),
            homographyMatrix
        )
        pointsTwoDimensional: np.ndarray = pointsParallelCalculated.reshape(-1, 2)
        return pointsTwoDimensional

    @staticmethod
    def GetVehicleColor(imageArray: np.ndarray, x: float, y: float, h: float, w: float) -> str:
        """获取车辆颜色"""
        imageHeight: int = imageArray.shape[0]
        imageWidth: int = imageArray.shape[1]

        x1: int = max(0, int(x - w / 2))
        y1: int = max(0, int(y - h / 2))
        x2: int = min(imageWidth, int(x + w / 2))
        y2: int = min(imageHeight, int(y + h / 2))

        if x2 <= x1 or y2 <= y1:
            return "unknown"

        fullRegionOfInterest: np.ndarray = imageArray[y1:y2, x1:x2]
        if fullRegionOfInterest.size == 0:
            return "unknown"

        def ClassifyRgbCenter(centerRgb: np.ndarray) -> str:
            """分类RGB中心颜色"""
            rgbPixel: np.ndarray = np.uint8([[centerRgb.astype(np.uint8)]])
            hsvPixel: np.ndarray = cv2.cvtColor(rgbPixel, cv2.COLOR_RGB2HSV)[0, 0]
            hueValue: int = int(hsvPixel[0])
            saturationValue: int = int(hsvPixel[1])
            valueValue: int = int(hsvPixel[2])
            if saturationValue < 55:
                if valueValue >= 200:
                    return "white"
                if valueValue <= 80:
                    return "black"
                return "gray"
            if (0 <= hueValue <= 12 or 155 <= hueValue <= 179) and saturationValue >= 60 and valueValue >= 60:
                return "red"
            if 14 <= hueValue <= 42 and saturationValue >= 65 and valueValue >= 60:
                return "yellow"
            if 92 <= hueValue <= 133 and saturationValue >= 60 and valueValue >= 50:
                return "blue"
            if valueValue >= 210:
                return "white"
            if valueValue <= 80:
                return "black"
            return "gray"

        def ClusterRegionOfInterest(regionOfInterest: np.ndarray) -> str:
            """聚类ROI颜色"""
            if regionOfInterest is None or regionOfInterest.size == 0:
                return "unknown"
            regionOfInterest = cv2.resize(regionOfInterest, (48, 48), interpolation=cv2.INTER_AREA)
            if regionOfInterest.shape[0] >= 3 and regionOfInterest.shape[1] >= 3:
                regionOfInterest = cv2.GaussianBlur(regionOfInterest, (3, 3), 0)
            regionOfInterestHSV: np.ndarray = cv2.cvtColor(regionOfInterest, cv2.COLOR_RGB2HSV)
            meanValue: float = np.mean(regionOfInterestHSV[:, :, 2])
            if meanValue < 60:
                return "black"
            pixels: np.ndarray = regionOfInterest.reshape(-1, 3).astype(np.float32)
            if len(pixels) > 5000:
                randomIndices: np.ndarray = np.random.choice(len(pixels), 5000, replace=False)
                pixels = pixels[randomIndices]
            if len(pixels) < 20:
                return "unknown"
            kMeansClusters: int = min(3, len(pixels))
            terminationCriteria: tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
            try:
                _, labels, centers = cv2.kmeans(pixels, kMeansClusters, None, terminationCriteria, 3, cv2.KMEANS_PP_CENTERS)
            except cv2.error:
                return "unknown"
            labels = labels.flatten()
            labelCounts: np.ndarray = np.bincount(labels, minlength=kMeansClusters)
            from collections import defaultdict
            colorVotes: defaultdict = defaultdict(float)
            for clusterIndex in range(kMeansClusters):
                classifiedColor: str = ClassifyRgbCenter(centers[clusterIndex])
                if classifiedColor != "unknown":
                    colorVotes[classifiedColor] += float(labelCounts[clusterIndex])
            if not colorVotes:
                dominantClusterIndex: int = int(np.argmax(labelCounts))
                return ClassifyRgbCenter(centers[dominantClusterIndex])
            bestColor: str = max(colorVotes, key=colorVotes.get)
            bestRatio: float = colorVotes[bestColor] / float(len(labels))
            if bestRatio < 0.08:
                return "unknown"
            return bestColor

        regionHeight: int = fullRegionOfInterest.shape[0]
        regionWidth: int = fullRegionOfInterest.shape[1]

        widthStart1: int = int(regionWidth * 0.16)
        widthEnd1: int = int(regionWidth * 0.84)
        heightStart1: int = int(regionHeight * 0.22)
        heightEnd1: int = int(regionHeight * 0.82)
        if widthEnd1 > widthStart1 and heightEnd1 > heightStart1:
            firstRegionOfInterest: np.ndarray = fullRegionOfInterest[heightStart1:heightEnd1, widthStart1:widthEnd1]
        else:
            firstRegionOfInterest = fullRegionOfInterest
        firstColor: str = ClusterRegionOfInterest(firstRegionOfInterest)
        if firstColor not in ("white", "black", "gray"):
            return firstColor

        widthStart2: int = int(regionWidth * 0.18)
        widthEnd2: int = int(regionWidth * 0.83)
        heightStart2: int = int(regionHeight * 0.07)
        heightEnd2: int = int(regionHeight * 0.52)
        if widthEnd2 > widthStart2 and heightEnd2 > heightStart2:
            secondRegionOfInterest: np.ndarray = fullRegionOfInterest[heightStart2:heightEnd2, widthStart2:widthEnd2]
        else:
            secondRegionOfInterest = fullRegionOfInterest
        secondColor: str = ClusterRegionOfInterest(secondRegionOfInterest)
        return firstColor if secondColor == "unknown" else secondColor

    @staticmethod
    def GetPositionPoint(xCoordinate: float, yCoordinate: float, height: float, width: float) -> tuple[float, float]:
        """获取位置点（底部中心）"""
        bottomX: float = xCoordinate
        bottomY: float = yCoordinate + height / 2
        return bottomX, bottomY

    @staticmethod
    def SegmentIntersect(pointA: tuple[float, float],
                         pointB: tuple[float, float],
                         pointP: tuple[float, float],
                         pointQ: tuple[float, float]) -> bool:
        """判断两条线段是否相交"""

        def CrossProduct(vector1: tuple[float, float], vector2: tuple[float, float]) -> float:
            return vector1[0] * vector2[1] - vector1[1] * vector2[0]

        def Subtract(vector1: tuple[float, float], vector2: tuple[float, float]) -> tuple[float, float]:
            return vector1[0] - vector2[0], vector1[1] - vector2[1]

        abVector: tuple[float, float] = Subtract(pointB, pointA)
        apVector: tuple[float, float] = Subtract(pointP, pointA)
        aqVector: tuple[float, float] = Subtract(pointQ, pointA)
        pbVector: tuple[float, float] = Subtract(pointB, pointP)
        pqVector: tuple[float, float] = Subtract(pointQ, pointP)
        paVector: tuple[float, float] = Subtract(pointA, pointP)
        crossProduct1: float = CrossProduct(abVector, apVector) * CrossProduct(abVector, aqVector)
        crossProduct2: float = CrossProduct(pqVector, paVector) * CrossProduct(pqVector, pbVector)
        return crossProduct1 <= 0 and crossProduct2 <= 0

    @staticmethod
    def PolygonArea(points: np.ndarray) -> float:
        """计算多边形面积"""
        xCoordinates: np.ndarray = points[:, 0]
        yCoordinates: np.ndarray = points[:, 1]
        areaResult: np.ndarray = 0.5 * abs(
            np.dot(xCoordinates, np.roll(yCoordinates, 1)) -
            np.dot(yCoordinates, np.roll(xCoordinates, 1))
        )
        return float(areaResult)

    @staticmethod
    def VehicleDetectionTransform(
            sourcePoints: np.ndarray,
            destinationPoints: np.ndarray,
            vehicleDetectionInformation: VehicleDetectionInformation
    ) -> VehicleStatisticsInformation:
        """车辆检测信息转换为统计信息"""
        trackIdentifier: int = vehicleDetectionInformation.trackIdentifier
        vehicleCategory: str = vehicleDetectionInformation.vehicleCategory
        frameIndexList: list[int] = []
        colorList: list[str] = []
        coordinateXList: list[float] = []
        coordinateYList: list[float] = []

        for informationIndex in range(len(vehicleDetectionInformation.frameIndexList)):
            frameIndex: int = vehicleDetectionInformation.frameIndexList[informationIndex]
            color: str = vehicleDetectionInformation.colorList[informationIndex]
            centerX: float = vehicleDetectionInformation.xCoordinateList[informationIndex]
            centerY: float = vehicleDetectionInformation.yCoordinateList[informationIndex]
            height: float = vehicleDetectionInformation.heightList[informationIndex]
            width: float = vehicleDetectionInformation.widthList[informationIndex]

            positionX, positionY = Tool.GetPositionPoint(
                xCoordinate=centerX,
                yCoordinate=centerY,
                height=height,
                width=width
            )
            pointArray: np.ndarray = np.array([[positionX, positionY]], dtype=np.float32)
            transformedPoints: np.ndarray = Tool.PerspectiveToParallelTransform(
                sourcePoints=sourcePoints,
                destinationPoints=destinationPoints,
                points=pointArray
            )
            frameIndexList.append(frameIndex)
            colorList.append(color)
            coordinateXList.append(float(transformedPoints[0][0]))
            coordinateYList.append(float(transformedPoints[0][1]))

        vehicleStatisticsInformation: VehicleStatisticsInformation = VehicleStatisticsInformation(
            trackIdentifier=trackIdentifier,
            vehicleCategory=vehicleCategory,
            frameIndexList=frameIndexList,
            colorList=colorList,
            coordinateXList=coordinateXList,
            coordinateYList=coordinateYList
        )
        return vehicleStatisticsInformation
