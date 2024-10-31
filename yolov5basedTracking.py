from zebrazoom.code.extractParameters import extractParameters
from types import SimpleNamespace

import cv2
import zebrazoom.videoFormatConversion.zzVideoReading as zzVideoReading
import numpy as np
import math
from zebrazoom.code.tracking import register_tracking_method
from ..._eyeTracking import EyeTrackingMixin
from ..._fasterMultiprocessingBase import BaseFasterMultiprocessing
from ..._getImages import GetImagesMixin

from zebrazoom.code.tracking.customTrackingImplementations.multipleCenterOfMassTracking._headTrackingHeadingCalculation import _headTrackingHeadingCalculation

import zebrazoom.code.tracking
import zebrazoom.code.tracking.customTrackingImplementations.yolov5.yolov5.detectZZ as detectZZ
from pathlib import Path


class Yolov5basedTracking(BaseFasterMultiprocessing, EyeTrackingMixin, GetImagesMixin):
  def __init__(self, videoPath, wellPositions, hyperparameters):
    super().__init__(videoPath, wellPositions, hyperparameters)
    if self._hyperparameters["eyeTracking"]:
      self._trackingEyesAllAnimalsList = [np.zeros((self._hyperparameters["nbAnimalsPerWell"], self._lastFrame-self._firstFrame+1, 8))
                                          for _ in range(self._hyperparameters["nbWells"])]
    else:
      self._trackingEyesAllAnimals = 0

    # if not(self._hyperparameters["nbAnimalsPerWell"] > 1) and not(self._hyperparameters["headEmbeded"]) and (self._hyperparameters["findHeadPositionByUserInput"] == 0) and (self._hyperparameters["takeTheHeadClosestToTheCenter"] == 0):
    self._trackingProbabilityOfGoodDetectionList = [np.zeros((self._hyperparameters["nbAnimalsPerWell"], self._lastFrame-self._firstFrame+1))
                                                      for _ in range(self._hyperparameters["nbWells"])]
    # else:
      # self._trackingProbabilityOfGoodDetectionList = 0
    
    if self._hyperparameters["headingCalculationMethod"]:
      self._trackingProbabilityOfHeadingGoodCalculation = [np.zeros((self._hyperparameters["nbAnimalsPerWell"], self._lastFrame-self._firstFrame+1)) for _ in range(self._hyperparameters["nbWells"])]
    else:
      self._trackingProbabilityOfHeadingGoodCalculation = 0

  def _adjustParameters(self, i, frame, widgets):
    return None

  def _formatOutput(self):
    if self._hyperparameters["postProcessMultipleTrajectories"]:
      for wellNumber in range(self._firstWell, self._lastWell + 1):
        self._postProcessMultipleTrajectories(self._trackingHeadTailAllAnimalsList[wellNumber], self._trackingProbabilityOfGoodDetectionList[wellNumber])
    
    if "postProcessHeadingWithTrajectory_minDist" in self._hyperparameters and self._hyperparameters["postProcessHeadingWithTrajectory_minDist"]:
      for wellNumber in range(self._firstWell, self._lastWell + 1):
        self._postProcessHeadingWithTrajectory(self._trackingHeadTailAllAnimalsList[wellNumber], self._trackingHeadingAllAnimalsList[wellNumber], self._trackingProbabilityOfHeadingGoodCalculation[wellNumber])
    
    return {wellNumber: extractParameters([self._trackingHeadTailAllAnimalsList[wellNumber], self._trackingHeadingAllAnimalsList[wellNumber], [], 0, 0] + ([self._auDessusPerAnimalIdList[wellNumber]] if self._auDessusPerAnimalIdList is not None else []), wellNumber, self._hyperparameters, self._videoPath, self._wellPositions, self._background)
            for wellNumber in range(self._firstWell, self._lastWell + 1)}

  def run(self):
    
    opt = SimpleNamespace(imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device='', view_img=False, save_txt=False, save_format=0, save_csv=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=Path('runs/detect'), name='exp', exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1, firstFrame=4, lastFrame=25, freqAlgoPosFollow=1)

    opt.weights = Path(self._hyperparameters["DLmodelPath"])
    opt.source  = Path(self._videoPath)
    opt.firstFrame = self._firstFrame
    opt.lastFrame  = self._lastFrame
    opt.freqAlgoPosFollow = self._hyperparameters["freqAlgoPosFollow"]
    
    res = detectZZ.main2(opt)
    
    wellNum = 0
    for frameNum in range(self._firstFrame, self._lastFrame):
      if frameNum < len(res):
        for animalNum in range(len(res[frameNum][0])):
          xCenter = float((res[frameNum][0][animalNum][0] + res[frameNum][0][animalNum][2]) / 2)
          yCenter = float((res[frameNum][0][animalNum][1] + res[frameNum][0][animalNum][3]) / 2)
          self._trackingHeadTailAllAnimalsList[wellNum][animalNum, frameNum-self._firstFrame][0][0] = int(xCenter)
          self._trackingHeadTailAllAnimalsList[wellNum][animalNum, frameNum-self._firstFrame][0][1] = int(yCenter)
    
    return self._formatOutput()


zebrazoom.code.tracking.register_tracking_method('yolov5basedTracking', Yolov5basedTracking)
