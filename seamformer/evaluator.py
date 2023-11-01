'''
Evaluate the JSON File 

- Line Accuracy (Algo):
* For every g belong to G , we find 1 p belonging to P and block the p for that list 
- Other Metrics : 
- There is no concept of best match anymore , we shift the perspective 
to prediction polygons matching to ground truth instead . 
* Pros : 
    1. It penalises multiple polygon predictions. 
    2. Scores will automatically degrade. 

- TextIoU integrated with BinaryNet 
- AveragePrecision @ IoU Thresholds

'''

# Libraries & Installations 
import sys
import copy 
import csv 
import json 
import os 
import cv2 
import argparse 
import numpy 
import numpy as np 

from medpy.metric import hd, hd95, assd
import seg_metrics.seg_metrics as sg

OUTDATED_IGNORE=1 

"""# Evaluator Class """
class Evaluator:
    def __init__(self,gds,preds,imgDims,img=None):
        self.gds=gds
        self.preds=preds
        
        self.binaryImage=None
        if img is not None:
            self.binaryImage = np.abs(255-img)
        self.H = imgDims[0]
        self.W = imgDims[1]
        self.image = np.zeros((self.H,self.W,1))
        self.iou_thresholds=None
        self.ap_dict = {}

        if self.iou_thresholds is not None:
            for iou in self.iou_thresholds:
                self.ap_dict[iou]=0.0
        
        self.pairs=self.findPairsV2()
       
    def reshapePolygons(self,polylist):
        plist=[]
        if len(polylist)>0:
            for gd in polylist:
                gd = np.asarray(gd,dtype=np.int32).reshape((-1,2))
                gd = gd.tolist()
                plist.append(gd)
        return plist
        
    def findPairsV2(self):
        '''
        For every prediction we find the correponding gd.
        Each polygon can map to same gd and get penalised.
        '''
        pairs = {}
        gds = copy.deepcopy(self.gds)
        preds = copy.deepcopy(self.preds)

        # Prediction Perspective !
        for i in range(0,len(preds)):
            predi = preds[i]
            maxscore = 0.0 
            index=None
            for j in range(0,len(gds)):
                gdj = gds[j]
                score,_ = self.iouPixelAcc(gdj,predi)
                if score is None:
                    continue 
                if score > maxscore :
                    maxscore = score 
                    index=j

            pairs[i]=index

        return pairs



    def lineAccuracyV2(self):
        '''
        Penalise if there is no match for gd polygon , that's all 
        '''
        N = len(self.gds)
        errors = 0 
        # Older version holds true only in this case 
        pairs = self.findPairs()
        for key,val in pairs.items():
            if val is None : 
                errors+=1
        lineAcc = 100*(np.abs(N-errors)/N)
        lineAcc= np.round(lineAcc,3)
        return lineAcc



    def findPairs(self):
        '''
        For every gt we find the corresponding prediction polygon based on area.
        '''
        pairs = {}
        taken = {}
        [taken.setdefault(x,True) for x in range(len(self.preds))]

        gds = copy.deepcopy(self.gds)
        preds = copy.deepcopy(self.preds)

        for i in range(0,len(gds)):
            gdi = gds[i]
            maxscore = 0.0
            index = None
            for j,p in enumerate(preds):
                score,_ = self.iouPixelAcc(gdi,p)
                if score is None :
                    continue 
                if score > maxscore and taken[j] is True:
                    maxscore = score 
                    index=j

            pairs[i]=index
            taken[index]=False

        return pairs
        

    def iouPixelAcc(self,gd,pred):
      if len(pred)<3:
          return 0.0
      else:
          # Create canvas 
          gcanvas = np.zeros(self.image.shape, dtype=np.uint8)
          pcanvas = np.zeros(self.image.shape, dtype=np.uint8)

          # GD - Prediction mask 
          gcanvas=cv2.fillPoly(gcanvas,pts=[np.asarray(gd,dtype=np.int32)],color=(255,255,255))
          pcanvas=cv2.fillPoly(pcanvas,pts=[np.asarray(pred,dtype=np.int32)],color=(255,255,255))
          
          # processing masks 
          intersection = cv2.bitwise_and(gcanvas,pcanvas)
          union = cv2.bitwise_or(gcanvas,pcanvas)

          # compute iou 
          iou = (intersection==255).sum() / float((union==255).sum())
          iou = np.round(iou,3)

          # compute acc
          total = (gcanvas==255).sum()
          correct = (intersection==255).sum()

          pixelacc = correct/total
          pixelacc= np.round(pixelacc,3)

          return iou,pixelacc
    

    def computeHDs(self):
        avgHDs=[]
        HDs=[]
        HD95s=[] 
        labels = [0,1]

        # Iterate through 
        for k,v in self.pairs.items():
            # Cannot compute HD,HD95,AvgHD if its not present 
            if v is None :
                continue
            else:
                gd = self.gds[v]
                pred = self.preds[k]

                gdCanvas = np.zeros((self.H,self.W,1))
                predCanvas = np.zeros((self.H,self.W,1))

                gd = np.asarray(gd,dtype=np.float32).reshape(-1,2)
                pred = np.asarray(pred,dtype=np.float32).reshape(-1,2)

                gdCanvas = cv2.fillPoly(gdCanvas,np.int32([gd]),1)
                predCanvas = cv2.fillPoly(predCanvas,np.int32([pred]),1)

                gdCanvas=gdCanvas.astype(np.uint8)
                predCanvas=predCanvas.astype(np.uint8)

                try : 
                    res_ahd, res_hd, res_hd95 = assd(predCanvas,gdCanvas), hd(predCanvas,gdCanvas), hd95(predCanvas,gdCanvas)
                    avgHDs.append(res_ahd)
                    HDs.append(res_hd)
                    HD95s.append(res_hd95)

                except Exception as exp: 
                    continue 
            
        # Average Computations 
        avgHDArr = np.array(avgHDs,dtype=np.float32)
        avgHDArr = avgHDArr[~numpy.isnan(avgHDArr)]

        HDArr = np.array(HDs,dtype=np.float32)
        HDArr =  HDArr[~numpy.isnan(HDArr)]

        HD95Arr = np.array(HD95s,dtype=np.float32)
        HD95Arr = HD95Arr[~numpy.isnan(HD95Arr)]
        

        # Net HD Parts 
        hdMean = np.round(np.mean(HDArr),3)
        avgHDMean = np.round(np.mean(avgHDArr),3)
        HD95Mean =np.round(np.mean(HD95Arr),3)

        # Score dictionary 
        evalDict = {'HD': hdMean,'AvgHD':avgHDMean,'HD95':HD95Mean}
        
        return evalDict

    
    def ioU(self):
        '''
        If there is a gd match , how well is it ?
        '''
        netIoUList=[]
        # For all the pairs 
        for k,v in self.pairs.items():
            if v is None :
                iou_k = 0.0
            else:
                iou_k,_ = self.iouPixelAcc(self.gds[v],self.preds[k])
            netIoUList.append(iou_k)
        
        netIoUList = np.asarray(netIoUList,dtype=np.float32)
        netIoUList = netIoUList[~numpy.isnan(netIoUList)]

        iouScore = np.mean(netIoUList)
        iouScore = np.round(iouScore,3)

        return  iouScore


    
    def pixelAcc(self):
        '''
        If there is a gd match , how correct is it ?
        '''
        netpixAccList=[]
        # For all the pairs 
        for k,v in self.pairs.items():
            if v is None :
                pixAcc_k=0.0
            else:
                gd = self.gds[v]
                pred = self.preds[k]
                _,pixAcc_k = self.iouPixelAcc(gd,pred)
            netpixAccList.append(pixAcc_k)
        
        netpixAccList = np.asarray(netpixAccList,dtype=np.float32)
        netpixAccList = netpixAccList[~numpy.isnan(netpixAccList)]

        pixelAccScore = np.mean(netpixAccList)
        pixelAccScore = np.round(pixelAccScore,3)
        stdpixelAccScore = np.round(np.std(netpixAccList),3)
        
        return pixelAccScore


    # Edited ou the score 
    def textIoU(self):
        '''
        Text Pixels Error Rate 
        -  Error Pixels / ( Correct Pixels of GD + Correct Pixels of Predicted Polygons )
        '''
        
        if  self.binaryImage is not None : 
            nettIouList = []
            # Key - PRED , VAL - GD
            for k,v in self.pairs.items():
                score=0.0
                if v is None :
                    nettIouList.append(0.0)
                    continue

                gd = self.gds[v]
                pred = self.preds[k]
                
                b1 = np.copy(self.binaryImage.copy())
                b2 = np.copy(self.binaryImage.copy())  
                
                # Empty Canvas 
                gcanvas = np.zeros(self.binaryImage.shape, dtype=np.uint8)
                pcanvas = np.zeros(self.binaryImage.shape, dtype=np.uint8)

                # Polygon Mask 
                gcanvas=cv2.fillPoly(gcanvas,pts=[np.asarray(gd,dtype=np.int32)],color=(255,255,255))
                pcanvas=cv2.fillPoly(pcanvas,pts=[np.asarray(pred,dtype=np.int32)],color=(255,255,255))

                
                # Grayscale convertion 
                pcanvas = cv2.cvtColor(pcanvas, cv2.COLOR_BGR2GRAY)
                gcanvas = cv2.cvtColor(gcanvas, cv2.COLOR_BGR2GRAY)

                # Pred - Binary Image Masking 
                b1[pcanvas==0] = 0
                b2[gcanvas==0] = 0

                # Convert them to maps 
                b1_gray=np.asarray(b1[:,:],dtype=np.int32)
                b2_gray=np.asarray(b2[:,:],dtype=np.int32)

                # Compute value 
                textErrorPixelMap = (b2_gray!=b1_gray)
                textErrorPixels = (textErrorPixelMap==True).sum()

                # Union pixels 
                unionPixelMap =(b1_gray==255)&(b2_gray==255)
                intersectPixelMap=(b2_gray==b1_gray)
                unionPixels = np.abs((unionPixelMap==True).sum()-(intersectPixelMap==True).sum())

                # Error / Union of maps 
                score = 1.0 - (textErrorPixels / float(unionPixels))                
                assert score<=1.0 and score>0.0 , 'Score is : {}  metric is wrong !'.format(str(score))
                
                nettIouList.append(score)

            if len(nettIouList)==0:
                return 0.0

            # Net Text Score 
            textiouScoreArr = np.asarray(nettIouList,dtype=np.float32)
            textiouScoreMean = np.round(np.mean(textiouScoreArr),3)   
            return textiouScoreMean

        else:
            return None 

    # For a single image 
    def get_single_image_results(self,iou_thresh):
        # Empty list of preds 
        all_pred_indices = range(len(self.preds))
        all_gt_indices = range(len(self.gds))
        if len(all_pred_indices) == 0:
            tp = 0
            fp = 0
            fn = len(self.gds)
            return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
        if len(all_gt_indices) == 0:
            tp = 0
            fp = len(self.preds)
            fn = 0
            return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
        
        # Matching process
        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for ipb, pred in enumerate(self.preds):
            for igb, gt in enumerate(self.gds):
                iou,_ = self.iouPixelAcc(gt,pred)
                if iou > iou_thresh:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)
      
        # Sort the IoU in descending order. 
        args_desc = np.argsort(ious)[::-1]
        if len(args_desc) == 0:
            # No matches
            tp = 0
            fp = len(self.preds)
            fn = len(self.gds)
        else:
            gt_match_idx = []
            pred_match_idx = []
            for idx in args_desc:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)

            tp = len(gt_match_idx)
            fp = len(self.preds) - len(pred_match_idx)
            fn = len(self.gds) - len(gt_match_idx)

        # Compute AvgPrecision & Avg Recall 
        true_pos = tp 
        false_pos = fp 
        false_neg =fn

        try:
            precision = true_pos/(true_pos + false_pos)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_pos/(true_pos + false_neg)
        except ZeroDivisionError:
            recall = 0.0

        return precision,recall

    # Generate AP@IoU>Threshold for a singe image 
    def AP_IoU(self):
        for iou_thresh in self.iou_thresholds:
            precision,recall = self.get_single_image_results(iou_thresh)
            self.ap_dict[iou_thresh]=precision


    # All scores !
    def computeAllScores(self):
        # la = self.lineAccuracyV2()
        la = 100.0
        iouScore = self.ioU()*100
        pixelAccScore = self.pixelAcc()
        combinedScore = self.computeHDs()
        if self.binaryImage is not None :
            textIoUScore = self.textIoU()
            finalDict = {'LineAcc':la,'pixelAcc':pixelAccScore,'IoU':iouScore,'HD':combinedScore['HD'],'AvgHD':combinedScore['AvgHD'],'HD95':combinedScore['HD95'],'textIoU':textIoUScore}
        else:
            finalDict = {'LineAcc':la,'pixelAcc':pixelAccScore,'IoU':iouScore,'HD':combinedScore['HD'],'AvgHD':combinedScore['AvgHD'],'HD95':combinedScore['HD95']}   
        return finalDict
