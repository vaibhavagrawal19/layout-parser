'''
Full Scale Inference : Stage I & II 
Input : Image Folder or Corresponding JSON File 
Output : JSON File : Image Information , Predicted Polygons & Scribbles 
'''

import json
import copy
import os 
import sys 
import csv 
import torch
import cv2
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from vit_pytorch.vit import ViT
from empatches import EMPatches
import argparse 
from torchvision.utils import make_grid
import utils_from_colab as pu

# Global settings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# File Imports 
from seam_conditioned_scribble_generation import *
from utils import *
from network import *

# Argument Parser 
def addArgs():
    # Required to override these params
    parser = argparse.ArgumentParser(description="SeamFormer:Inference")
    parser.add_argument("--exp_name",type=str, help="Unique Experiment Name",required=True,default=None)
    parser.add_argument("--input_image_folder",type=str, help="Input Folder Path",default=None)
    parser.add_argument("--input_image_json",type=str, help="Input JSON Path",required=False,default=None)
    parser.add_argument("--output_image_folder",type=str, help="Output Folder Path for storing bin & scr results",required=True,default=None)
    parser.add_argument("--model_weights_path",type=str,help="Model Checkpoint Weights",default=None)
    parser.add_argument("--input_json", action="store_true", help="Inference Based on JSON File ")
    parser.add_argument("--input_folder", action="store_true", help="Inference Based on Image Folder")
    parser.add_argument("--vis", action="store_true", help="Visualisation Flag")
    
    # Fixed Arguments ( override in special cases only)
    parser.add_argument("--encoder_layers",type=int, help="Encoder Level Layers",default=6)
    parser.add_argument("--encoder_heads",type=int, help="Encoder Heads",default=8)
    parser.add_argument("--encoder_dims",type=int, help="Internal Encoder Dim",default=768)
    parser.add_argument("--img_size",type=int, help="Image Shape",default=256)
    parser.add_argument("--patch_size",type=int, help="Input Patch Shape",default=8)
    parser.add_argument("--split_size",type=int, help="Splitting Image Dim",default=256)
    parser.add_argument("--threshold",type=float,help="Prediction Thresholding",default=0.40)
    parser.add_argument("--bin_flag",type=int, help="To get binarised input for URDU",default=0)
    parser.add_argument("--model_weights_path_bin",type=str,help="Model Checkpoint Weights for binarisation",default="../seamformer/weights/I2.pt")
    args_ = parser.parse_args()
    settings = vars(args_)
    return settings

'''
Takes in the default settings 
and args to create the network.
'''
# Network Configuration  
def buildModel(settings):
    print('Present here : {}'.format(settings))
    # Encoder settings
    encoder_layers = settings['encoder_layers']
    encoder_heads = settings['encoder_heads']
    encoder_dim = settings['encoder_dims']
    patch_size = settings['patch_size']
    # Encoder
    v = ViT(
        image_size = settings['img_size'],
        patch_size =  settings['patch_size'],
        num_classes = 1000,
        dim = encoder_dim,
        depth = encoder_layers,
        heads = encoder_heads,
        mlp_dim = 2048)
    
    # Full model
    network =  SeamFormer(encoder = v,
        decoder_dim = encoder_dim,      
        decoder_depth = encoder_layers,
        decoder_heads = encoder_heads,
        patch_size = patch_size)
    
    print('Model Weight Loading ...')
    # Load pre-trained network + letting the encoder network also trained in the process.
    if settings['model_weights_path'] is not None:
        if ('Arabic' in settings['model_weights_path']) and (settings['bin_flag']):
            try:
                network.load_state_dict(torch.load(settings['model_weights_path_bin'], map_location=device),strict=True)
                print('Network Weights loaded successfully!')
            except Exception as exp :
                print('Network Weights Loading Error , Exiting !: %s' % exp)
                sys.exit()
            settings['bin_flag'] = 0

        elif os.path.exists(settings['model_weights_path']):
            try:
                network.load_state_dict(torch.load(settings['model_weights_path'], map_location=device),strict=True)
                print('Network Weights loaded successfully!')
            except Exception as exp :
                print('Network Weights Loading Error , Exiting !: %s' % exp)
                sys.exit()
        else:
            print('Network Weights File Not Found')
            sys.exit()

    network = network.to(device)
    network.eval()
    return network


'''
Performs both binary and scribble output generation.
'''
def imageInference(network,path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True):
    emp = EMPatches()
    if not os.path.exists(path):
        print('Exiting! Invalid Image Path : {}'.format(path))
        sys.exit(0)
    else:
        weight = torch.tensor(1) #Dummy weight 
        parentImage=cv2.imread(path)
        input_patches , indices = readFullImage(path,PDIM,DIM,OVERLAP)

        patch_size=args['patch_size']
        img_size = args['img_size']
        spilt_size = args['split_size']
        image_size = (spilt_size,spilt_size)
        THRESHOLD = args['threshold']

        soutput_patches=[]
        boutput_patches=[]
        # Iterate through the resulting patches
        for i,sample in enumerate(input_patches):
            p = sample['img']
            target_shape = (sample['resized'][1],sample['resized'][0])
            with torch.no_grad():
                inputs =torch.from_numpy(p).to(device)
                # Pass through model
                loss_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight, reduction='none')
                pred_pixel_values_bin,pred_pixel_values_scr=network(inputs,gt_bin_img=inputs,gt_scr_img=inputs,criterion=loss_criterion,strain=True,btrain=True,mode='test')

                # Send them to .cpu
                pred_pixel_values_bin = pred_pixel_values_bin.cpu()
                pred_pixel_values_scr = pred_pixel_values_scr.cpu()

                bpatch=reconstruct(pred_pixel_values_bin,patch_size,target_shape,image_size)
                spatch=reconstruct(pred_pixel_values_scr,patch_size,target_shape,image_size)

                # binarize the predicted image taking 0.5 as threshold
                bpatch = ( bpatch>THRESHOLD)*1
                spatch = ( spatch>THRESHOLD)*1

                # Append the net processed patch
                soutput_patches.append(255*spatch)
                boutput_patches.append(255*bpatch)

        try:
            assert len(boutput_patches)==len(soutput_patches)==len(input_patches)
        except Exception as exp:
            print('Error in patch processing outputs : Exiting!')
            sys.exit(0)
        
        # Restich the image
        soutput = emp.merge_patches(soutput_patches,indices,mode='max')
        boutput = emp.merge_patches(boutput_patches,indices,mode='max')

        # Transpose
        binaryOutput=np.transpose(boutput,(1,0))
        scribbleOutput=np.transpose(soutput,(1,0))
        
        return binaryOutput,scribbleOutput
    
'''
Post Processing Function
'''
def postProcess(scribbleImage,binaryImage,binaryThreshold=50,rectangularKernel=50):
    bin_ = binaryImage.astype(np.uint8)
    scr = scribbleImage.astype(np.uint8)
    # print('PP @ BIN SHAPE : {} SCRIBBLE SHAPE : {}'.format(scribbleImage.shape,binaryImage.shape))
    # bin_ = cv2.cvtColor(bin_,cv2.COLOR_BGR2GRAY)
    H,W = bin_.shape

    # Threshold it
    bin_[bin_>=binaryThreshold]=255
    bin_[bin_<binaryThreshold]=0
    scr[scr>=binaryThreshold]=255
    scr[scr<binaryThreshold]=0

    # We apply distance transform to thin the output polygon
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    scr = polygon_to_distance_mask(scr,threshold=30)

    # Bitwise AND of the textual region and polygon region ( only cut off letters will be highlighted)
    scr_ = cv2.bitwise_and(bin_/255,scr/255)
    # Dilate the existing text content 
    # scr_ = text_dilate(scr_,kernel_size=3,iterations=3) # SD = 3,3 
    scr_ = text_dilate(scr_,kernel_size=3,iterations=7) # KH 3,7

    # Dilate it horizontally to fill the gaps within the text region 
    # scr_ = horizontal_dilation(scr_,rectangularKernel,3) # SD - 50 ,3 
    scr_ = horizontal_dilation(scr_,rectangularKernel,3) # KH - 50 ,2 
   
    # Extract the final contours 
    scr_ = np.repeat(scr_[:, :, np.newaxis], 3, axis=2) 
    contours = cleanImageFindContours(scr_,threshold = 0.15)
    # Combine the hulls that are on the same horizontal level 
    new_hulls = combine_hulls_on_same_level(contours)
    # Scribble Generation
    predictedScribbles=[]
    for hull in new_hulls:
        hull = np.asarray(hull,dtype=np.int32).reshape((-1,2)).tolist()
        scr_ = generateScribble(H,W,hull)
        if scr_ is not None:
            scr_ = np.asarray(hull).reshape((-1,2)).tolist()
            predictedScribbles.append(scr_)
    return predictedScribbles

def postProcess_BKS(scribbleImage,binaryImage,binaryThreshold=40,rectangularKernel=30):
    scr = np.repeat(scribbleImage[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(binaryImage[:, :, np.newaxis], 3, axis=2)
    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)
    # print('PP @ BIN SHAPE : {} SCRIBBLE SHAPE : {}'.format(scribbleImage.shape,binaryImage.shape))
    # bin_ = cv2.cvtColor(bin_,cv2.COLOR_BGR2GRAY)
    H,W,_ = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # We apply distance transform to thin the output polygon
    
    tmp = polygon_to_distance_mask(scr,threshold=30)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    # Threshold it
    bin[bin>=binaryThreshold]=255
    bin[bin<binaryThreshold]=0
    scr[scr>=binaryThreshold]=255
    scr[scr<binaryThreshold]=0

    # Bitwise AND of the textual region and polygon region ( only cut off letters will be highlighted)
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    # Dilate the existing text content 
    # scr_ = text_dilate(scr_,kernel_size=3,iterations=3) # SD = 3,3 
    scr_ = text_dilate(scr_,kernel_size=3,iterations=3) # KH 3,7

    # Dilate it horizontally to fill the gaps within the text region 
    # scr_ = horizontal_dilation(scr_,rectangularKernel,3) # SD - 50 ,3 
    scr_ = horizontal_dilation(scr_,rectangularKernel,1) # KH - 50 ,2 
   
    # Extract the final contours 
    contours = cleanImageFindContours(scr_,threshold = 0.10)

    # Combine the hulls that are on the same horizontal level 
    new_hulls = combine_hulls_on_same_level(contours, tolerance=30)
    
    # Scribble Generation
    predictedScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        contours = cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        predictedScribbles.append(scr_lst)
    return predictedScribbles


# Post Process Function for both penn, jain+asr
def postProcess_combined(scr, bin,thresh=40, rectangularKernel=7):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)
    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)
    tmp = pu.polygon_to_distance_mask(scr,threshold=60)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0

    # cv2_imshow(255*scr)

    # Usage of binarization results
    kernel = np.ones((2, 2), np.uint8)
    scr = cv2.erode(scr, kernel, iterations=1) # even if gaps are produced, merge hulls will take care of it
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=3,iterations=3)
    scr_ = pu.horizontal_dilation(scr_, rectangularKernel,1)

    box_img, box, cut = pu.find_text_bounding_box(scr_*255) # use binarisation output for

    scr_ = pu.get_subset(box, scr_)

    contours = pu.cleanImageFindContours(scr_,threshold = 0.10)
    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # # Hull operation - merge contours
    # canvas = np.zeros(scr_.shape)

    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=20)
    genScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy, cut+cut//2 ,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)

    return genScribbles
def post_processing_I2_JAIN_ASR(scr, bin, thresh=40, rectangularKernel=50):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)
    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)

    tmp = pu.polygon_to_distance_mask(scr,threshold=60)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0

    # Usage of binarization results
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=3,iterations=3)
    scr_ = pu.horizontal_dilation(scr_, rectangularKernel, 2)

    box_img, box = pu.get_box(scr_.copy(), cut=460)

    scr_ = pu.get_subset(box, scr_)

    # temporary contours
    contours = pu.cleanImageFindContours(scr_,threshold = 0.10)

    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # Hull operation - merge contours
    canvas = np.zeros(scr_.shape)
    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=30)
    genScribbles=[]
    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy,390,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)

    return genScribbles


def post_processing_I2_PENN(scr, bin, thresh=40, rectangularKernal=7):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)
    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)
    tmp = pu.polygon_to_distance_mask(scr,threshold=60)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0


    # Usage of binarization results
    kernel = np.ones((2, 2), np.uint8)  
    scr = cv2.erode(scr, kernel, iterations=2) # even if gaps are produced, merge hulls will take care of it
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=3,iterations=3)

    scr_ = pu.horizontal_dilation(scr_, 7,1)

    box_img, box, cut = pu.find_text_bounding_box(scr_*255) # use binarisation output for 

    scr_ = pu.get_subset(box, scr_)

    # temporary contours
    contours = pu.cleanImageFindContours(scr_,threshold = 0.10)

    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # # Hull operation - merge contours
    # canvas = np.zeros(scr_.shape)

    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=10)
    genScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy, 2*cut ,1)
        # canvas_copy = horizontal_dilation(canvas_copy,200,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)

    return genScribbles
def postProcess_KG(scr, bin, thresh=40, rectangularKernel=70):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)

    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)

    tmp = pu.polygon_to_distance_mask(scr,threshold=40)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0

    # Usage of binarization results
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=2,iterations=2)
    scr_ = pu.horizontal_dilation(scr_,rectangularKernel,1)

    # box_img, box, cut = pu.find_text_bounding_box(scr_*255, factor=0.25)
    # scr_ = pu.get_subset(box, scr_)
    box_img, box = pu.get_box(scr_.copy(), cut=460)
    scr_ = pu.get_subset(box, scr_)

    # temporary contours
    contours = pu.cleanImageFindContours(scr_,threshold = 0.05)
    contour_image = np.zeros_like(scr_)
    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # Hull operation - merge contours
    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=16)
    genScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy,390,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)
    return genScribbles

def postProcess_URDU(scr, bin, thresh=40, rectangularKernel=30):
    scr = np.repeat(scr[:, :, np.newaxis], 3, axis=2) 
    bin = np.repeat(bin[:, :, np.newaxis], 3, axis=2)

    bin = bin.astype(np.uint8)
    scr = scr.astype(np.uint8)

    tmp = pu.polygon_to_distance_mask(scr,threshold=30)
    final_tmp = np.zeros_like(scr)
    for j in range(3):
        final_tmp[:, :, j] = tmp
    scr = final_tmp
    bin[bin>=thresh]=255
    bin[bin<thresh]=0
    scr[scr>=thresh]=255
    scr[scr<thresh]=0

    # Usage of binarization results
    scr_ = cv2.bitwise_and(bin/255,scr/255)
    scr_ = pu.dilate_image(scr_,kernel_size=3,iterations=2)
    scr_ = pu.horizontal_dilation(scr_,rectangularKernel,1)

    box_img, box, cut = pu.find_text_bounding_box(scr_*255)

    scr_ = pu.get_subset(box, scr_)

    # temporary contours
    contours = pu.cleanImageFindContours(scr_,threshold = 0.10)

    H,W,C = bin.shape
    mask_with_contours=copy.deepcopy(bin)

    # Hull operation - merge contours
    _,new_hulls = pu.combine_hulls_on_same_level(contours, threshold=30)
    genScribbles=[]

    for c in new_hulls  :
        canvas_copy = np.zeros(scr_.shape)
        c = np.asarray(c,dtype=np.int32).reshape((-1,1,2))
        canvas_copy = cv2.fillPoly(canvas_copy,np.int32([c]),(255,255,255))

        canvas_copy = pu.horizontal_dilation(canvas_copy,cut + cut//2,1)

        contours = pu.cleanImageFindContours(canvas_copy,threshold = 0.10)
        h=np.asarray(contours[0],dtype=np.int32)
        h = cv2.convexHull(h)
        h=np.asarray(h,dtype=np.int32).reshape((-1,2))
        h=h.tolist()
        scr = pu.generateScribble(H,W,h)
        scr_arr = np.asarray(scr,dtype=np.int32).reshape((-1,1,2))
        mask_with_contours=cv2.polylines(mask_with_contours,[scr_arr], isClosed=False, color=(0,255,0),thickness=3)

        scr_lst = scr_arr.reshape((-1,2)).tolist()
        genScribbles.append(scr_lst)
    return genScribbles

'''
Perform Binary inference for URDU
'''
def Bin_Inference(args):
    # Get the model first 
    network = buildModel(args)
    print('Completed loading weight')
    # Make output directory if its not present 
    bin_folder =  os.path.join(args['input_image_folder'], 'bin')
    os.makedirs(bin_folder,exist_ok=True)

    files_ = os.listdir(args['input_image_folder'])
    if len(files_)>0:
        for f in files_ : 
            try:
                print('Processing image - {}'.format(f))
                file_path = os.path.join(args['input_image_folder'],f)
                file_name = os.path.basename(file_path)
                img = cv2.imread(file_path)
                H,W,C = img.shape
                
                # Calling Stage I Inference 
                binaryMap,scribbleMap =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
                binaryMap=np.uint8(binaryMap)

                cv2.imwrite(os.path.join(bin_folder,file_name),binaryMap)

            except Exception as exp:
                print('Image :{} Error :{}'.format(file_name,exp))
                continue

    else:
        print('Empty Input Image Folder , Exiting !')
        sys.exit(0)

    print('~Completed Inference !')   


'''
Performs Binary & Scribble 
Inference given imageFolder
'''

def Inference(args):
    # Get the model first 
    network = args["model_weights"]
    # print('Completed loading weight')

    # Make output directory if its not present 
    os.makedirs(args['output_image_folder'],exist_ok=True)
    vis_folder = None
    scr_folder = None
    bin_folder = None
    # Make a seperate scribble & binary image folders 
    if args["bin_vis"]:
        bin_folder =  os.path.join(args['output_image_folder'],'bin')
        os.makedirs(bin_folder,exist_ok=True)
    if args["scr_vis"]:
        scr_folder = os.path.join(args['output_image_folder'],'scr')
        os.makedirs(scr_folder,exist_ok=True)
    if args["poly_vis"]:
        vis_folder =  os.path.join(args['output_image_folder'],'vis')
        os.makedirs(vis_folder,exist_ok=True)

    # # Case I : Input JSON 
    # if args['input_image_json'] is not None and os.path.exists(args['input_image_json']):
    #     # Read and open the input json file 
    #     with open(args['input_image_json'], "r") as json_file:
    #         data = json.load(json_file)
    #     print('Evaluating {} samples ..'.format(len(data)))
    #     jsonOutput = []

    #     for i,record in enumerate(data):
    #         try:
    #             print('Processing image - {}'.format(record['imgPath']))
    #             file_path = record['imgPath']
    #             file_name = os.path.basename(file_path)
    #             img = cv2.imread(file_path)
    #             H,W,C = img.shape

    #             # Model Inference..
    #             binaryMap,scribbleMap =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
    #             binaryMap=np.uint8(binaryMap)
    #             scribbleMap=np.uint8(scribbleMap)

    #             if 'BKS' in args['model_weights_path']:
    #                 scribbles = postProcess_BKS(scribbleMap,binaryMap,binaryThreshold=50,rectangularKernel=50)
    #             elif 'I2_PENN' in args['model_weights_path']:
    #                 scribbles = post_processing_I2_PENN(scribbleMap,binaryMap)
    #             elif 'I2' in args['model_weights_path']:
    #                 scribbles = post_processing_I2_JAIN_ASR(scribbleMap,binaryMap,thresh=40,rectangularKernel=50)
    #             elif 'Arabic' in args['model_weights_path']:
    #                 scribbles = postProcess_URDU(scribbleMap,imgMap)
    #             else:
    #                 scribbles =  postProcess_combined(scribbleMap,binaryMap,thresh=40, rectangularKernel=7)
                 
    #             # Storing .. # Put it under vis flag 
    #             cv2.imwrite(os.path.join(scr_folder,'scr_'+file_name),scribbleMap)
    #             cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)

    #             # Sending to Stage 2
    #             binaryMap = cv2.imread(os.path.join(bin_folder,'bin_'+file_name))
    #             scribbleMap = cv2.imread(os.path.join(scr_folder,'scr_'+file_name))
            
    #             ppolygons = imageTask(img,binaryMap,scribbles)

    #             # Visualise the ppolygons once 
    #             img2 = copy.deepcopy(img)
    #             for p in ppolygons:
    #                 p = np.asarray(p,dtype=np.int32).reshape((-1,1,2))
    #                 img2 = cv2.polylines(img2, [p],True, (255, 0, 0),3)
    #             cv2.imwrite(os.path.join(vis_folder,'vis_'+file_name),img2)

    #             # Writing it to JSON file 
    #             scrs_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in scribbles]
    #             pps_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in ppolygons]

    #             jsonOutput.append({'imgPath':record['imgPath'],'imgDims':[H,W],'predScribbles':scrs_,'predPolygons':pps_})
             
    #         except Exception as exp:
    #             print('Image :{} Error :{}'.format(file_name,exp))
    #             continue
    
    # Case II : Image Folder 
    
    files_ = os.listdir(args['input_image_folder'])
    jsonOutput = []
    annotated_imgs = []
    all_images_polygon_list = []
    if len(files_)>0:
        for f in files_ : 
            # try:
            print('Processing image - {}'.format(f))
            if 'Arabic' in args['model_weights_path']:
                file_path = os.path.join(args['input_image_folder'],'bin',f)
            else:
                file_path = os.path.join(args['input_image_folder'],f)
            file_name = os.path.basename(file_path)
            img = cv2.imread(file_path)
            H,W,C = img.shape
            
            start_time = time.time()
            # Calling Stage I Inference 
            binaryMap,scribbleMap =  imageInference(network,file_path,args,PDIM=256,DIM=256,OVERLAP=0.25,save=True)
            binaryMap=np.uint8(binaryMap)
            scribbleMap=np.uint8(scribbleMap)
            imgMap = np.uint8(img)[:,:,0]

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Stage 1 output took {elapsed_time:.4f} seconds to run.")

            start_time = time.time()
            # Post Processing of Scribble Branch
            if 'BKS' in args['model_weights_path']:
                scribbles = postProcess_BKS(scribbleMap,binaryMap,binaryThreshold=50,rectangularKernel=50)
            elif 'I2_PENN' in args['model_weights_path']:
                scribbles = post_processing_I2_PENN(scribbleMap,binaryMap)
            elif 'I2' in args['model_weights_path']:
                scribbles = post_processing_I2_JAIN_ASR(scribbleMap,binaryMap,thresh=40,rectangularKernel=50)
            elif 'Arabic' in args['model_weights_path']:
                scribbles = postProcess_URDU(scribbleMap,imgMap)
            else:
                scribbles = postProcess_combined(scribbleMap,binaryMap,thresh=40, rectangularKernel=7)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Stage 1 Postprocessing took {elapsed_time:.4f} seconds to run.")
            
            if args["scr_vis"]:
                cv2.imwrite(os.path.join(scr_folder,'scr_'+file_name),scribbleMap)
            if args["bin_vis"]:
                # cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)
                if 'Arabic' in args['model_name']:
                    cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),imgMap)
                else:
                    cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)

            # # Visualisation purpose
            # cv2.imwrite(os.path.join(scr_folder,'scr_'+file_name),scribbleMap)
            # if 'Arabic' in args['model_weights_path']:
            #     cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),imgMap)
            # else:
            #     cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)

            # Preparation for Stage II 
            # binaryMap = cv2.imread(os.path.join(bin_folder,'bin_'+file_name))
            if 'Arabic' in args['model_name']:
                # cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),imgMap)
                binaryMap = np.repeat(imgMap[:, :, None], 3, axis=-1)
            else:
                # cv2.imwrite(os.path.join(bin_folder,'bin_'+file_name),binaryMap)
                binaryMap = np.repeat(imgMap[:, :, None], 3, axis=-1)
                
            # scribbleMap = cv2.imread(os.path.join(scr_folder,'scr_'+file_name))
            
            start_time = time.time()
            # Stage II Output : Text Line Polygons 
            ppolygons = imageTask(img,binaryMap,scribbles)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Stage 2 took {elapsed_time:.4f} seconds to run.")
        
            # Visualise the predicted polygons and store it 
            print(file_path)
            if 'Arabic' in args['model_weights_path']:
                print(file_path.replace(r'/bin', ''))
                img2 = cv2.imread(file_path.replace(r'/bin', ''))
            else:
                img2 = copy.deepcopy(img)
            
            for p in ppolygons:
                p = np.asarray(p,dtype=np.int32).reshape((-1,1,2))
                img2 = cv2.polylines(img2, [p],True, (255, 0, 0),3)
            cv2.imwrite(os.path.join(vis_folder,'vis_'+file_name),img2)

            # Writing it to JSON file 
            scrs_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in scribbles]
            pps_ = [ np.asarray(gd).reshape((-1,2)).tolist() for gd in ppolygons]

            jsonOutput.append({'imgPath':file_path,'imgDims':[H,W],'predScribbles':scrs_,'predPolygons':pps_})
            print()
            # except Exception as exp:
            # print('Image :{} Error :{}'.format(file_name,exp))
            # continue


    # Save the json file 
    with open(os.path.join(args['output_image_folder'],'{}.json'.format(args['exp_name'])),'w') as f:
        json.dump(jsonOutput,f)
    f.close() 
    print('~Completed Inference !')   

if __name__ == "__main__":    
    args = addArgs()
    print('Running Inference...')
    if 'Arabic' in args['model_weights_path']:
        start_time = time.time()
        args['bin_flag'] = 1
        Bin_Inference(args)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function 'BIN_inference' took {elapsed_time:.4f} seconds to run.")

    Inference(args)

