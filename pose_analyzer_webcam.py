import os
import numpy as np
import cv2
import subprocess as sp
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import matplotlib.pyplot as plt
from visualize_webcam import *
import time
import torch.multiprocessing as mp
import shutil
import subprocess as sp
import math


def get_img_paths(imgs_dir):
	img_paths = []
	for dirpath, dirnames, filenames in os.walk(imgs_dir):
		for filename in [f for f in filenames if f.endswith('.png') or f.endswith('.PNG') or f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.jpeg') or f.endswith('.JPEG')]:
			img_paths.append(os.path.join(dirpath,filename))
	img_paths.sort()

	return img_paths

def read_images(dir_path):
	img_paths = get_img_paths(dir_path)
	for path in img_paths:
		yield cv2.imread(path)


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)


def read_video(filename):
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def init_pose_predictor(config_path, weights_path, cuda=True):
	cfg = get_cfg()
	cfg.merge_from_file(config_path)
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	cfg.MODEL.WEIGHTS = weights_path
	if cuda == False:
		cfg.MODEL.DEVICE='cpu'
	predictor = DefaultPredictor(cfg)

	return predictor


def encode_for_videpose3d(boxes,keypoints,resolution, dataset_name):
	# Generate metadata:
	metadata = {}
	metadata['layout_name'] = 'coco'
	metadata['num_joints'] = 17
	metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
	metadata['video_metadata'] = {dataset_name: resolution}

	prepared_boxes = []
	prepared_keypoints = []



	prepared_boxes.append(boxes)
	prepared_keypoints.append(keypoints[:,:2])
		
	boxes = np.array(prepared_boxes, dtype=np.float32)
	keypoints = np.array(prepared_keypoints, dtype=np.float32)
	keypoints = keypoints[:, :, :2] # Extract (x, y)
	
	# Fix missing bboxes/keypoints by linear interpolation
	mask = ~np.isnan(boxes[:, 0])
	indices = np.arange(len(boxes))
	for i in range(4):
		boxes[:, i] = np.interp(indices, indices[mask], boxes[mask, i])
	for i in range(17):
		for j in range(2):
			keypoints[:, i, j] = np.interp(indices, indices[mask], keypoints[mask, i, j])
	
	#print('{} total frames processed'.format(len(boxes)))
	#print('{} frames were interpolated'.format(np.sum(~mask)))
	#print('----------')
	
	return [{
		'start_frame': 0, # Inclusive
		'end_frame': len(keypoints), # Exclusive
		'bounding_boxes': boxes,
		'keypoints': keypoints,
	}], metadata


def predict_pose(pose_predictor, img_generator, output_path, dataset_name='detectron2'):
	'''
		pose_predictor: The detectron's pose predictor
		img_generator:  Images source
		output_path:    The path where the result will be saved in .npz format
	'''
	boxes = []
	keypoints = []
	resolution = None

	# Predict poses:
	for i, img in enumerate(img_generator):
		pose_output = pose_predictor(img)

		if len(pose_output["instances"].pred_boxes.tensor) > 0:
			cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
			cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
		else:
			cls_boxes = np.full((4,), np.nan, dtype=np.float32)
			cls_keyps = np.full((17,3), np.nan, dtype=np.float32)   # nan for images that do not contain human

		boxes.append(cls_boxes)
		keypoints.append(cls_keyps)

		# Set metadata:
		if resolution is None:
			resolution = {
				'w': img.shape[1],
				'h': img.shape[0],
			}

		print('{}      '.format(i+1), end='\r')

	# Encode data in VidePose3d format and save it as a compressed numpy (.npz):
	data, metadata = encode_for_videpose3d(boxes, keypoints, resolution, dataset_name)
	output = {}
	output[dataset_name] = {}
	output[dataset_name]['custom'] = [data[0]['keypoints'].astype('float32')]

	#np.savez_compressed(output_path, positions_2d=output, metadata=metadata)
    #print()
	print (output[dataset_name]['custom'])
    




def predict_pose_webcam(pose_predictor, img, dataset_name='detectron2'):
    '''
    	pose_predictor: The detectron's pose predictor
    	img_generator:  Images source
    	output_path:    The path where the result will be saved in .npz format
    '''
    boxes = []
    keypoints = []
    resolution = None   

    #rsz_img = cv2.resize(img, (640, 360))
    # Predict poses:
    #starttime = time.time()
    pose_output = pose_predictor(img)   
    #endtime = time.time()
    #tottime = endtime - starttime
    if len(pose_output["instances"].pred_boxes.tensor) > 0:
    	cls_boxes = pose_output["instances"].pred_boxes.tensor[0].cpu().numpy()
    	cls_keyps = pose_output["instances"].pred_keypoints[0].cpu().numpy()
    else:
    	cls_boxes = np.full((4,), np.nan, dtype=np.float32)
    	cls_keyps = np.full((17,3), np.nan, dtype=np.float32)   # nan for images that do not contain human  
    # boxes.append(cls_boxes)
    # keypoints.append(cls_keyps) 
    # Set metadata:
    if resolution is None:
    	resolution = {
    		'w': img.shape[1],
    		'h': img.shape[0],
    	}   
    
    # Encode data in VidePose3d format and save it as a compressed numpy (.npz):
    data, metadata = encode_for_videpose3d(cls_boxes, cls_keyps, resolution, dataset_name)
    output = {}
    output[dataset_name] = {}
    output[dataset_name]['custom'] = [data[0]['keypoints'].astype('float32')]   
    #np.savez_compressed(output_path, positions_2d=output, metadata=metadata)
    #print()
    #print (output[dataset_name]['custom'])
    #print("total time", tottime)
    return output









if __name__ == '__main__':
	# Init pose predictor:
	model_config_path = '/home/moonbeam/anaconda3/envs/TFCVenv/lib/python3.6/site-packages/detectron2/model_zoo/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'
	model_weights_path = '/home/moonbeam/Desktop/Galvanize/Capstone/Projects/FinalProject-03/ComputerVision/DeepLearningMachine/VideoPose3d_with_Detectron2/models/model_final_5ad38f.pkl'	
	pose_predictor = init_pose_predictor(model_config_path, model_weights_path, cuda=True)  
	# Predict poses and save the result:
	img_generator = read_images('imgs')    # read images from a directory
	#img_generator = read_video('./video.mp4')  # or get them from a video
	img = cv2.imread('imgs/Lusquat_001.jpg')
	print(type(img))
	output_path = 'predictions/Lusquatpred_001' 
	start = time.time()
	
	#predict_pose_webcam(pose_predictor, img)    
	end = time.time()   
	elapsed_time = end-start    
	# print(elapsed_time)	
	# get the reference to the webcam
	CAMERA = cv2.VideoCapture(0)
	# CAMERA.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	# CAMERA.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)	
	while(True):
		# read a new frame
		start = time.time()
		check, frame = CAMERA.read()    
		# show the frame
		#  plt.imshow(frame)
		#  plt.show()		
		keyps = predict_pose_webcam(pose_predictor, frame)
		#Visualize the keypoints:
		visualize_keypoints(frame, keyps)       
		end = time.time()   
		elapsed_time = end-start    
		print("frames per second:", 1/elapsed_time)     
		cv2.imshow("Capturing frames", frame)       
		# quit camera if 'q' key is pressed
		if cv2.waitKey(10) & 0xFF == ord("q"):
		    break		
	CAMERA.release()
	cv2.destroyAllWindows()

   

