import os
import shutil
import numpy as np
import cv2
import subprocess as sp
import math
from math import atan




def load_keypoints_from_npz(path_to_npz, dataset_name='detectron2'):
	data = np.load(path_to_npz, encoding='latin1', allow_pickle=True)
	meta = data['metadata'].item()
	keypoints = data['positions_2d'].item()[dataset_name]['custom'][0]

	return keypoints, meta


def remove_dir(dir):
	try:
		shutil.rmtree(dir)
	except OSError as e:
		print ("Error: %s - %s." % (e.filename, e.strerror))


def frames_to_video(src_path, dst_path, fps=30):
	os.system("ffmpeg -framerate %s -pattern_type glob -f image2 -i '%s/*.jpeg' %s" % (fps, src_path, dst_path))


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
		yield cv2.imread(path), path


def read_video(filename):
	w, h = get_resolution(filename)

	command = ['ffmpeg',
			'-i', filename,
			'-f', 'image2pipe',
			'-pix_fmt', 'bgr24',
			'-vsync', '0',
			'-vcodec', 'rawvideo', '-']

	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	i = 0
	while True:
		i += 1
		data = pipe.stdout.read(w*h*3)
		if not data:
			break
		yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3)), str(i-1).zfill(5)


def get_resolution(filename):
	command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
			   '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
	pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
	for line in pipe.stdout:
		w, h = line.decode().strip().split(',')
		return int(w), int(h)


def draw_body_joints_2d(img_orig, pts2d, bones=None, draw_indices=None):
    img = img_orig
    #print(pts2d['detectron2']['custom'][0][0][0][0])
    #"right_shoulder"   7
    #"right_elbow"      9    
    #"right_hip"        13

    for i in range(len(pts2d)):
        if not math.isnan(pts2d['detectron2']['custom'][0][0][0][0]) and (not math.isnan(pts2d['detectron2']['custom'][0][0][0][1])):
            x = int(round(pts2d['detectron2']['custom'][0][0][0][0]))
            y = int(round(pts2d['detectron2']['custom'][0][0][0][1]))
            cv2.circle(img, (x,y), 5, (0,0,255), 5)
            if draw_indices is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (x,y)
                fontScale = 1
                fontColor = (255,255,255)
                lineType = 2
                cv2.putText(img,str(i), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
            if bones is not None:
                for bone in bones:
                    pt1 = (int(round(pts2d['detectron2']['custom'][0][0][bone[0]][0])), int(round(pts2d['detectron2']['custom'][0][0][bone[0]][1])))
                    pt2 = (int(round(pts2d['detectron2']['custom'][0][0][bone[1]][0])), int(round(pts2d['detectron2']['custom'][0][0][bone[1]][1])))
                    cv2.line(img,pt1,pt2,(255,0,0),4)
                    cv2.circle(img, pt1, 5, (0,0,255), 5)
        #print(pts2d['detectron2']['custom'][0][0])


        #Right shoulder
        angle_pt1 = pts2d['detectron2']['custom'][0][0][6]
        r_shoulder_pt1_x = angle_pt1[0]
        r_shoulder_pt1_y = angle_pt1[1]                    
        #Right elbow
        angle_pt2 = pts2d['detectron2']['custom'][0][0][8]
        r_elbow_pt2_x = angle_pt2[0]
        r_elbow_pt2_y = angle_pt2[1]                
        #Right hip
        angle_pt3 = pts2d['detectron2']['custom'][0][0][12]
        r_hip_pt3_x = angle_pt3[0]
        r_hip_pt3_y = angle_pt3[1]                  
        shoulder_elbow_angle = atan(abs(r_shoulder_pt1_x - r_elbow_pt2_x)/ abs(r_shoulder_pt1_y - r_elbow_pt2_y))
        shoulder_hip_angle = atan(abs(r_shoulder_pt1_x - r_hip_pt3_x)/ abs(r_shoulder_pt1_y - r_hip_pt3_y))             
        shoulder_anti_flexion = (shoulder_elbow_angle + shoulder_hip_angle)*180/math.pi
        #print("shoulder angle: ", shoulder_anti_flexion)
        rotate = 90 - (shoulder_hip_angle*180/math.pi)          
        cv2.ellipse(img,(r_shoulder_pt1_x, r_shoulder_pt1_y),(50,50), rotate, 0, shoulder_anti_flexion,(40, 200, 150),-1)
        color = (0, 255, 0)
        thickness = 5
        start_pt = (r_shoulder_pt1_x, r_shoulder_pt1_y)
        end_pt = (r_elbow_pt2_x, r_elbow_pt2_y)
        cv2.arrowedLine(img, start_pt, end_pt, color, thickness=thickness)    
                
                
    


def visualize_keypoints(frame, keypoints, draw_joint_indices=None):
    '''
    Visualize keypoints (2d body joints) detected by Detectron2:
    	img_generator:      Images source (images or video)
    	keypoints:          Body keypoints detected by Detectron2
    	mp4_output_path:    The path where the result will be saved in .mp4 format
    	fps:                FPS of the result video
    	draw_joint_indices: Draw body joint indices (in COCO format)
    '''
    body_edges_17 = np.array([[0,1],[1,3],[2,0],[4,2],[5,7],[6,5],[7,9],[8,6],[10,8],
    						  [11,5],[12,6],[12,11],[13,11],[14,12],[15,13],[16,14]])   
    #Create a temp_dir to save intermediate results:
    # temp_dir = './temp'
    # if os.path.exists(temp_dir):
    # 	remove_dir(temp_dir)
    # os.makedirs(temp_dir) 
    #Draw keypoints and save the result:
    frame_joint2d = keypoints

    #print(frame_joint2d)

    draw_body_joints_2d(frame, frame_joint2d, bones=body_edges_17, draw_indices=draw_joint_indices) 
    	#img_name = img_path.split('/')[-1].split('.')[0]
    	#out_path = os.path.join(temp_dir,img_name + '.jpeg')
    	#cv2.imwrite(out_path,img)  
    	#print('{}      '.format(i+1), end='\r')    
    # #Convert images to video:
    # frames_to_video(temp_dir, mp4_output_path, fps=fps)
    # remove_dir(temp_dir)  
    # print ('All done!')
    


