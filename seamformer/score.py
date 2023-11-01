''' 
JSON Evaluator 
'''

import os 
import cv2
import wandb
import json
import csv
import numpy as np
import argparse
import math


# File Imports 
from Evaluator import Evaluator

def argumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datacodes',type=str,default='I2')
    parser.add_argument('--expdir', type=str,default='/home/niharika.v/VIS',required=True)
    parser.add_argument('--expName', type=str,default='test_01',required=True)
    parser.add_argument('--projectName', type=str,default='StageII-Testing')
    parser.add_argument('--outputJsonPath',type=str,required=True)
    args = parser.parse_args()
    # WandB Login 
    print('Logging into WandB Project...')
    wandb.init(project=args.projectName,id=args.expName,resume='allow')
    return args

def remove_nans(lst):
    cleaned_list = [x for x in lst if not math.isnan(x)]
    return cleaned_list

def score(args):
    print('Json Path : {}'.format(args.outputJsonPath))
    with open(args.outputJsonPath,'r') as f:
        res = json.load(f)
    print('Loaded : {}'.format(len(res)))
    # Listed Metrics 
    evalDict={'IoU':[],'HD':[],'AvgHD':[],'HD95':[]}
    # Timing Metrics 
    timeDict={'H':[],'W':[],'time':[]}    
    samples=0
    # Iterate over results
    # res=[res[0]]
    for i,pt in enumerate(res):
        try:
            print('Computing score : {}'.format(pt['imgPath']))
            gds = pt['gdPolygons']
            predpolys = pt['predPolygons']
            #time = int(float(pt['time']))
            #print('Time : {} {}'.format(time,type(time)))
            imgDims = pt['imgDims']
            H,W = imgDims
        
            eval = Evaluator(gds,predpolys,imgDims,None)
            score_i = eval.computeAllScores()
            
            if predpolys is not None:
                # Evaluation of metrics
                for k in evalDict.keys():
                    if score_i[k] is not None and not np.isnan(score_i[k]): 
                        evalDict[k].append(score_i[k])
                # Timing Analysis
                timeDict['H'].append(np.int32(H))
                timeDict['W'].append(np.int32(W))
                #timeDict['time'].append(np.int32(time))
                samples+=1
        except Exception as exp: 
            print("Error in Score Computation : {}".format(exp))
            continue
    
    # WandB Visualization

    # Bar Plot
    scoreDict={'IoU':0.0,'HD':0.0,'AvgHD':0.0,'HD95':0.0}
    for k,v in evalDict.items():
        if len(v) == 1 :
            scoreDict[k] = np.round(v[0],3)
        else:
            v = remove_nans(v)
            v = np.asarray(v,dtype=np.float32)
            v = v[~np.isnan(v)]
            scoreDict[k]=np.round(np.mean(v),3)
        wandb.log({k:scoreDict[k]})

    metricData=[]
    for k,v in scoreDict.items():
        metricData.append([k,v])
    table = wandb.Table(data=metricData, columns=["Metrics", "Value"])
    plot = wandb.plot.bar(table, "metrics","value", title="Score")
    wandb.log({"score_bar_plots":plot})
    wandb.plot.bar(table, "metrics","value", title="Score")

    # Manually post the logs onto a separate text file in the output folder 
    with open(os.path.join(args.expdir,'{}_{}_EVALSCORE.csv'.format(args.datacodes,args.expName)),'w') as f:
        writer = csv.writer(f)
        for key, value in scoreDict.items():
            writer.writerow([key, value])
        f.close()
    print('Saved Score Card ..!')

    # Done 
    print('~Completed')

if __name__ == '__main__':
    args=argumentParser()
    print('Starting evaluation..')
    score(args)
    print('~Completed')
