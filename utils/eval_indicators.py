
"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""

import cv2
import os
import numpy as np
from scipy.io import loadmat
__all__ = ['SegmentationMetric']

"""
confusionMetric  
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def OA(self):
        # Overall accuracy The ratio of correct pixels to total pixels
        #  OA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def UA(self):
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc
    def PA(self):
        #PA=TP/(TP+FN)
        classacc=np.diag(self.confusionMatrix)/self.confusionMatrix.sum(axis=1)
        return classacc
    def Kappa(self):
        "The kappa coefficient is suitable for situations where the number of samples in each category is not balanced. In this case, an indicator that can penalize the bias of the model is needed to replace acc."
        po=self.OA()
        pe_rows=self.confusionMatrix.sum(axis=0)
        pe_cols=self.confusionMatrix.sum(axis=1)
        sumtotal=self.confusionMatrix.sum()
        pe=np.dot(pe_rows,pe_cols)/float(sumtotal**2)
        return (po-pe)/(1-pe)


    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy (MPA): This is a simple improvement on PA. It calculates the proportion of correctly classified pixels in each class and then takes the average of all classes.
        :return:
        """
        classAcc = self.UA()
        meanAcc = classAcc[classAcc < float('inf')].mean()
        return meanAcc

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.IntersectionOverUnion()
        mIoU = IoU[IoU<float('inf')].mean()
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel, ignore_labels):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        for IgLabel in ignore_labels:
            mask &= (imgLabel != IgLabel)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU, Frequency Weighted Intersection Over Union: An improvement over MIoU, this method weights each class according to its frequency of occurrence.
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel, ignore_labels):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel, ignore_labels)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def F1_score(self,Class):
        pc=self.confusionMatrix[Class,Class]/np.sum(self.confusionMatrix,0)[Class]
        re=self.confusionMatrix[Class,Class]/np.sum(self.confusionMatrix,1)[Class]
        F1=2*pc*re/(pc+re)
        return F1




