import uuid
from typing import List
import shutil

import cv2
from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

# from ...app import load_seamformer_models

from .dependencies import save_uploaded_images
# from .helper import (process_image, process_image_craft,
#                      process_image_worddetector, process_multiple_image_craft,
#                      process_multiple_image_doctr,
#                      process_multiple_image_doctr_v2,
#                      process_multiple_image_worddetector, save_uploaded_image)
from .models import LayoutImageResponse, ModelChoice, SeamFormerResponse, SeamFormerChoice, SeamFormerArgs, PolygonModel, PolygonList, AllImagesPolygonList
# from .post_helper import process_dilate, process_multiple_dilate
#
from seamformer.inference import Inference

# File Imports for SeamFormer
from seamformer.seam_conditioned_scribble_generation import *
from seamformer.utils import *
from seamformer.network import *
from vit_pytorch.vit import ViT

i2_weights = None
bks_weights = None
i2kg_weights = None
i2penn_weights = None
urdu_weights = None

router = APIRouter(
	prefix='/layout',
	tags=['Main'],
)

def buildModel(settings, weights_path):
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
	
	# print('Model Weight Loading ...')
	# Load pre-trained network + letting the encoder network also trained in the process.
	network =  SeamFormer(encoder = v,
		decoder_dim = encoder_dim,      
		decoder_depth = encoder_layers,
		decoder_heads = encoder_heads,
		patch_size = patch_size)

	if weights_path is not None:
		if os.path.exists(weights_path):
			try:
				network.load_state_dict(torch.load(weights_path, map_location=device),strict=True)
				# print('Network Weights loaded successfully!')
			except Exception as exp :
				print('Network Weights Loading Error , Exiting !: %s' % exp)
				sys.exit()
		else:
			print('Network Weights File Not Found')
			sys.exit()

	network = network.to(device)
	network.eval()
	return network

def load_seamformer_models():
	settings = {
		"encoder_layers": 6,
		"encoder_heads": 8,
		"encoder_dims": 768,
		"img_size": 256,
		"patch_size": 8,
		"split_size": 256,
		"threshold": 0.30,
	}
	global i2_weights
	global bks_weights
	global i2kg_weights
	global i2penn_weights
	global urdu_weights
	i2_weights = buildModel(settings, "./seamformer/I2.pt")
	bks_weights = buildModel(settings, "./seamformer/BKS.pt")
	i2kg_weights = buildModel(settings, "./seamformer/I2_KG.pt")
	i2penn_weights = buildModel(settings, "./seamformer/I2_PENN.pt")
	urdu_weights = buildModel(settings, "./seamformer/Arabic.pt")

# load_seamformer_models()

# @router.post('/seamformer/debug', response_model=List[SeamFormerResponse])
@router.post('/seamformer/debug')
def debug_and_visualize_intermediate_outputs_for_Seamformer(
	folder_path: str = Depends(save_uploaded_images),
	args_json: SeamFormerArgs = Depends(),
	model_variant: SeamFormerChoice = Form(SeamFormerChoice.I2),
	):
	shutil.rmtree("output")
	if os.path.exists("output.zip"):
		os.remove("output.zip")
	global i2_weights
	global bks_weights
	global i2kg_weights
	global i2penn_weights
	global urdu_weights
	"""
	API endpoint for calling seamformer-layout-parser
	""" 
	# if i2_weights is not None:
	#     print(f"success!")
	# else:
	#     print(f"failure")
	print(f"folder_path: {folder_path}")
	model_weights = None
	model_name = None
	if model_variant == SeamFormerChoice.I2:
		model_weights = i2_weights
		model_name = "I2"
	elif model_variant == SeamFormerChoice.BKS:
		model_weights = bks_weights
		model_name = "BKS"
	elif model_variant == SeamFormerChoice.I2_PIH:
		model_weights = i2penn_weights
		model_name = "I2_PIH"
	elif model_variant == SeamFormerChoice.I2_KG:
		model_weights = i2kg_weights
		model_name = "I2_KG"
	elif model_variant == SeamFormerChoice.URDU:
		model_weights = urdu_weights
		model_name = "Arabic"

	args = {
		"mode": "debug",
		"exp_name": "v0",
		"input_image_folder": folder_path,
		"output_image_folder": "./output",
		"model_weights": model_weights,
		"i2_weights": i2_weights,
		"input_folder": True,
		"encoder_layers": 6,
		"encoder_heads": 8,
		"encoder_dims": 768,
		"img_size": 256,
		"patch_size": 8,
		"split_size": 256,
		"threshold": 0.30,
		"bin_vis": args_json.visualize_binarized,
		"scr_vis": args_json.visualize_scribbles,
		"poly_vis": args_json.visualize_polygons,
		"poly_json": args_json.json_output,
		"model_name": model_name
	}

	print(f"poly_json is {args_json.json_output}")

	Inference(args)
	os.system("zip -r output.zip output")
	shutil.rmtree("output")
	os.mkdir("output")
	return FileResponse("output.zip", filename="output.zip", headers={"Content-Disposition": f"attachment; filename=output.zip"})


@router.post('/seamformer/visualize')
def visualize_the_final_output_containing_annotated_polygons(
	folder_path: str = Depends(save_uploaded_images),
	model_variant: SeamFormerChoice = Form(SeamFormerChoice.I2),
	):
	shutil.rmtree("./output")
	global i2_weights
	global bks_weights
	global i2kg_weights
	global i2penn_weights
	global urdu_weights
	"""
	API endpoint for calling seamformer-layout-parser
	""" 
	# if i2_weights is not None:
	#     print(f"success!")
	# else:
	#     print(f"failure")
	print(f"folder_path: {folder_path}")
	model_weights = None
	model_name = None
	if model_variant == SeamFormerChoice.I2:
		model_weights = i2_weights
		model_name = "I2"
	elif model_variant == SeamFormerChoice.BKS:
		model_weights = bks_weights
		model_name = "BKS"
	elif model_variant == SeamFormerChoice.I2_PIH:
		model_weights = i2penn_weights
		model_name = "I2_PIH"
	elif model_variant == SeamFormerChoice.I2_KG:
		model_weights = i2kg_weights
		model_name = "I2_KG"
	elif model_variant == SeamFormerChoice.URDU:
		model_weights = urdu_weights
		model_name = "Arabic"

	args = {
		"mode": "visualize",
		"exp_name": "v0",
		"input_image_folder": folder_path,
		"output_image_folder": "./output",
		"model_weights": model_weights,
		"i2_weights": i2_weights,
		"input_folder": True,
		"encoder_layers": 6,
		"encoder_heads": 8,
		"encoder_dims": 768,
		"img_size": 256,
		"patch_size": 8,
		"split_size": 256,
		"threshold": 0.30,
		"bin_vis": False,
		"scr_vis": False,
		"poly_vis": True,
		"poly_json": False,
		"model_name": model_name
	}

	Inference(args)
	filename = os.listdir("./output/vis/")[0]
	return FileResponse(f"./output/vis/{filename}")


@router.post('/seamformer/predict')
def get_a_JSON_containing_predicted_polygons(
	folder_path: str = Depends(save_uploaded_images),
	model_variant: SeamFormerChoice = Form(SeamFormerChoice.I2),
	):
	shutil.rmtree("output")
	global i2_weights
	global bks_weights
	global i2kg_weights
	global i2penn_weights
	global urdu_weights
	"""
	API endpoint for calling seamformer-layout-parser
	""" 
	# if i2_weights is not None:
	#     print(f"success!")
	# else:
	#     print(f"failure")
	print(f"folder_path: {folder_path}")
	model_weights = None
	model_name = None
	if model_variant == SeamFormerChoice.I2:
		model_weights = i2_weights
		model_name = "I2"
	elif model_variant == SeamFormerChoice.BKS:
		model_weights = bks_weights
		model_name = "BKS"
	elif model_variant == SeamFormerChoice.I2_PIH:
		model_weights = i2penn_weights
		model_name = "I2_PIH"
	elif model_variant == SeamFormerChoice.I2_KG:
		model_weights = i2kg_weights
		model_name = "I2_KG"
	elif model_variant == SeamFormerChoice.URDU:
		model_weights = urdu_weights
		model_name = "Arabic"

	args = {
		"mode": "debug",
		"exp_name": "v0",
		"input_image_folder": folder_path,
		"output_image_folder": "./output",
		"model_weights": model_weights,
		"i2_weights": i2_weights,
		"input_folder": True,
		"encoder_layers": 6,
		"encoder_heads": 8,
		"encoder_dims": 768,
		"img_size": 256,
		"patch_size": 8,
		"split_size": 256,
		"threshold": 0.30,
		"bin_vis": False,
		"scr_vis": False,
		"poly_vis": False,
		"poly_json": True,
		"model_name": model_name
	}

	points = Inference(args)
	# return FileResponse("./output/v0.json")
	return FileResponse("./output/v0.json", filename="v0.json", headers={"Content-Disposition": f"attachment; filename=v0.json"})
	# return all_images_polygon_list
	# points = [
	#     [[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]],
	#     [[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]],
	#     [[[1, 2], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]]
	# ]
	# image_models = []
	# for image_result in points:
	# 	polygon_models = []
	# 	for polygon in image_result:
	# 		polygon_models.append(PolygonModel(points=polygon))
	# 	image_models.append(PolygonList(polygons=polygon_models))
	# return AllImagesPolygonList(image_polygons=image_models)
	# all_images_polygon_list_model = []
	# for polygon_list in all_images_polygon_list:
	#     polygon_list_model = []
	#     for polygon in polygon_list:
	#         print(len(polygon))
	#         print(len(polygon[0]))
	#         polygon_model = IntPairsModel(pairs=polygon)
	#         polygon_list_model.append(polygon_model)
	#     # polygon_list_model = PolygonList(polygons=polygon_list_model)
	#     all_images_polygon_list_model.append(polygon_list_model)
	# return AllImagesPolygonList(image_polygons=all_images_polygon_list_model)



# @router.post('/', response_model=List[LayoutImageResponse])
# async def doctr_layout_parser(
# 	folder_path: str = Depends(save_uploaded_images),
# 	model: ModelChoice = Form(ModelChoice.doctr),
# 	dilate: bool = Form(False),
# ):
# 	"""
# 	API endpoint for calling the layout parser
# 	"""
# 	print(model.value)
# 	if model == ModelChoice.craft:
# 		ret = process_multiple_image_craft(folder_path)
# 	elif model == ModelChoice.worddetector:
# 		ret = process_multiple_image_worddetector(folder_path)
# 	elif model == ModelChoice.doctr:
# 		ret = process_multiple_image_doctr(folder_path)
# 	elif model == ModelChoice.v2_doctr:
# 		ret = process_multiple_image_doctr_v2(folder_path)
# 	if dilate:
# 		ret = process_multiple_dilate(ret)
# 	return ret

#
# @router.post('/visualize')
# async def layout_parser_swagger_only_demo(
# 	image: UploadFile = File(...),
# 	model: ModelChoice = Form(ModelChoice.doctr),
# 	dilate: bool = Form(False),
# ):
# 	"""
# 	This endpoint is only used to demonstration purposes.
# 	this endpoint returns/displays the input image with the
# 	bounding boxes clearly marked in blue rectangles.
#
# 	PS: This endpoint is not to be called from outside of swagger
# 	"""
# 	image_path = save_uploaded_image(image)
# 	if model == ModelChoice.craft:
# 		regions = process_image_craft(image_path)
# 	elif model == ModelChoice.worddetector:
# 		regions = process_image_worddetector(image_path)
# 	else:
# 		regions = process_image(image_path, model.value)
# 	if dilate:
# 		regions = process_dilate(regions, image_path)
# 	save_location = '/home/layout/layout-parser/images/{}.jpg'.format(
# 		str(uuid.uuid4())
# 	)
# 	# TODO: all the lines after this can be transfered to the helper.py file
# 	bboxes = [i.bounding_box for i in regions]
# 	bboxes = [((i.x, i.y), (i.x+i.w, i.y+i.h)) for i in bboxes]
# 	img = cv2.imread(image_path)
# 	count = 1
# 	for i in bboxes:
# 		img = cv2.rectangle(img, i[0], i[1], (0,0,255), 3)
# 		img = cv2.putText(
# 			img,
# 			str(count),
# 			(i[0][0]-5, i[0][1]-5),
# 			cv2.FONT_HERSHEY_COMPLEX,
# 			1,
# 			(0,0,255),
# 			1,
# 			cv2.LINE_AA
# 		)
# 		count += 1
# 	cv2.imwrite(save_location, img)
# 	return FileResponse(save_location)
