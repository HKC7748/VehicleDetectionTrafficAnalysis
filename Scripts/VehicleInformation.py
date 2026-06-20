import numpy as np
import statistics
from scipy.interpolate import interp1d


def KalmanFilterTrajectory(frameTimeArray: np.ndarray,
                           coordinateXArray: np.ndarray,
                           coordinateYArray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """卡尔曼滤波轨迹平滑"""
    numberOfObservations: int = len(frameTimeArray)
    deltaTime: float = 1.0
    stateTransitionMatrix: np.ndarray = np.array([
        [1, 0, deltaTime, 0],
        [0, 1, 0, deltaTime],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    measurementMatrix: np.ndarray = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)
    processNoiseValue: float = 0.001
    processNoiseCovariance: np.ndarray = np.eye(4, dtype=float) * processNoiseValue
    processNoiseCovariance[2:, 2:] *= 0.1
    measurementNoiseValue: float = 5.0
    measurementNoiseCovariance: np.ndarray = np.eye(2, dtype=float) * measurementNoiseValue

    if numberOfObservations >= 2:
        initialVelocityX: float = (coordinateXArray[1] - coordinateXArray[0]) / (frameTimeArray[1] - frameTimeArray[0])
        initialVelocityY: float = (coordinateYArray[1] - coordinateYArray[0]) / (frameTimeArray[1] - frameTimeArray[0])
    else:
        initialVelocityX, initialVelocityY = 0.0, 0.0

    estimatedState: np.ndarray = np.array([
        coordinateXArray[0],
        coordinateYArray[0],
        initialVelocityX,
        initialVelocityY
    ], dtype=float)

    estimatedCovariance: np.ndarray = np.eye(4, dtype=float) * 10.0
    filteredCoordinateXArray: np.ndarray = np.zeros(numberOfObservations, dtype=float)
    filteredCoordinateYArray: np.ndarray = np.zeros(numberOfObservations, dtype=float)
    velocityMagnitudeArray: np.ndarray = np.zeros(numberOfObservations, dtype=float)

    for observationIndex in range(numberOfObservations):
        predictedState: np.ndarray = stateTransitionMatrix @ estimatedState
        predictedCovariance: np.ndarray = (stateTransitionMatrix @ estimatedCovariance @ stateTransitionMatrix.T + processNoiseCovariance)
        observationVector: np.ndarray = np.array([coordinateXArray[observationIndex], coordinateYArray[observationIndex]], dtype=float)
        measurementResidual: np.ndarray = (observationVector - measurementMatrix @ predictedState)
        residualCovariance: np.ndarray = (measurementMatrix @ predictedCovariance @ measurementMatrix.T + measurementNoiseCovariance)
        kalmanGain: np.ndarray = (predictedCovariance @ measurementMatrix.T @ np.linalg.inv(residualCovariance))
        estimatedState = predictedState + kalmanGain @ measurementResidual
        estimatedCovariance = (np.eye(4, dtype=float) - kalmanGain @ measurementMatrix) @ predictedCovariance
        filteredCoordinateXArray[observationIndex] = estimatedState[0]
        filteredCoordinateYArray[observationIndex] = estimatedState[1]
        velocityMagnitudeArray[observationIndex] = np.sqrt(estimatedState[2] ** 2 + estimatedState[3] ** 2)

    return filteredCoordinateXArray, filteredCoordinateYArray, velocityMagnitudeArray


class VehicleDetectionInformation:
    """车辆检测信息类，存储单辆车的检测数据"""

    def __init__(self,
                 trackIdentifier: int,
                 vehicleCategory: str,
                 frameIndex: int,
                 color: str,
                 probability: float,
                 xCoordinate: float,
                 yCoordinate: float,
                 width: float,
                 height: float) -> None:
        self.trackIdentifier: int = trackIdentifier
        self.vehicleCategory: str = vehicleCategory
        self.frameIndexList: list[int] = [frameIndex]
        self.colorList: list[str] = [color]
        self.probabilityList: list[float] = [probability]
        self.xCoordinateList: list[float] = [xCoordinate]
        self.yCoordinateList: list[float] = [yCoordinate]
        self.widthList: list[float] = [width]
        self.heightList: list[float] = [height]

    def AddFrameInformation(self,
                            frameIndex: int,
                            color: str,
                            probability: float,
                            xCoordinate: float,
                            yCoordinate: float,
                            width: float,
                            height: float) -> None:
        """添加帧信息"""
        self.frameIndexList.append(frameIndex)
        self.colorList.append(color)
        self.probabilityList.append(probability)
        self.xCoordinateList.append(xCoordinate)
        self.yCoordinateList.append(yCoordinate)
        self.widthList.append(width)
        self.heightList.append(height)


class VehicleStatisticsInformation:
    """车辆统计信息类，存储单辆车的拟合轨迹和速度"""

    def __init__(self,
                 trackIdentifier: int,
                 vehicleCategory: str,
                 frameIndexList: list[int],
                 colorList: list[str],
                 coordinateXList: list[float],
                 coordinateYList: list[float]) -> None:
        self.trackIdentifier: int = trackIdentifier
        self.vehicleCategory: str = vehicleCategory
        self.dominantColor: str = statistics.mode(colorList)
        self.frameIndexList: list[int] = frameIndexList
        self.colorList: list[str] = colorList
        self.color: str = statistics.mode(self.colorList)
        self.coordinateXList: list[float] = coordinateXList
        self.coordinateYList: list[float] = coordinateYList
        self.fittedFrameIndexList: list[int] = []
        self.fittedCoordinateXList: list[float] = []
        self.fittedCoordinateYList: list[float] = []
        self.estimatedVelocityList: list[float] = []

    def Update(self) -> None:
        """更新拟合轨迹和速度"""
        self.fittedFrameIndexList.clear()
        self.fittedCoordinateXList.clear()
        self.fittedCoordinateYList.clear()
        self.estimatedVelocityList.clear()

        if len(self.frameIndexList) < 15:
            return

        frameTimeArray: np.ndarray = np.array(self.frameIndexList, dtype=float)
        coordinateXArray: np.ndarray = np.array(self.coordinateXList, dtype=float)
        coordinateYArray: np.ndarray = np.array(self.coordinateYList, dtype=float)
        filteredCoordinateXArray: np.ndarray
        filteredCoordinateYArray: np.ndarray
        velocityMagnitudeArray: np.ndarray

        filteredCoordinateXArray, filteredCoordinateYArray, velocityMagnitudeArray = KalmanFilterTrajectory(frameTimeArray, coordinateXArray, coordinateYArray)

        fittedFrameIndexArray: np.ndarray = np.arange(self.frameIndexList[0], self.frameIndexList[-1] + 30 + 1, 1)

        interpolationFunctionX: interp1d = interp1d(frameTimeArray, filteredCoordinateXArray, kind='linear', fill_value='extrapolate')
        interpolationFunctionY: interp1d = interp1d(frameTimeArray, filteredCoordinateYArray, kind='linear', fill_value='extrapolate')
        interpolationFunctionVelocity: interp1d = interp1d(frameTimeArray, velocityMagnitudeArray, kind='linear', fill_value='extrapolate')

        self.fittedFrameIndexList = fittedFrameIndexArray.tolist()
        fittedCoordinateXArray: np.ndarray = interpolationFunctionX(fittedFrameIndexArray)
        self.fittedCoordinateXList = fittedCoordinateXArray.tolist()
        fittedCoordinateYArray: np.ndarray = interpolationFunctionY(fittedFrameIndexArray)
        self.fittedCoordinateYList = fittedCoordinateYArray.tolist()
        estimatedVelocityArray: np.ndarray = interpolationFunctionVelocity(fittedFrameIndexArray)
        estimatedVelocityArray: np.ndarray = np.abs(estimatedVelocityArray)
        self.estimatedVelocityList = estimatedVelocityArray.tolist()

    def GetPositionInFrame(self, frameIndex: int) -> tuple[float, float] | None:
        """获取指定帧的位置"""
        if frameIndex not in self.fittedFrameIndexList:
            return None
        indexPosition: int = self.fittedFrameIndexList.index(frameIndex)
        return self.fittedCoordinateXList[indexPosition], self.fittedCoordinateYList[indexPosition]
