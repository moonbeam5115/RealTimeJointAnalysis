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


def draw_joint_angles(img, pivot_pt, distal_pt1, distal_pt2, color=(40, 200, 150)):
    r_shoulder_x = pivot_pt[0]
    r_shoulder_y = pivot_pt[1]

    r_elbow_x = distal_pt1[0]
    r_elbow_y = distal_pt1[1]

    r_hip_x = distal_pt2[0]
    r_hip_y = distal_pt2[1]
    
    shoulder_elbow_angle = atan(abs(r_shoulder_x - r_elbow_x)/ abs(r_shoulder_y - r_elbow_y))
    shoulder_hip_angle = atan(abs(r_shoulder_x - r_hip_x)/ abs(r_shoulder_y - r_hip_y))             
    shoulder_anti_flexion = (shoulder_elbow_angle + shoulder_hip_angle)*180/math.pi
    #print("shoulder angle: ", shoulder_anti_flexion)
    rotate = 90 - (shoulder_hip_angle*180/math.pi)          
    cv2.ellipse(img,(r_shoulder_x, r_shoulder_y),(50,50), rotate, 0, shoulder_anti_flexion,color,-1)

    return shoulder_anti_flexion


def draw_joint_reaction_force(img, joint, ground_contact, mid_joint, opp_ground_contact, color = (0, 255, 0), thickness = 5):
    
    #Biomechanics/Physics Calculations
    adductor_angle = 71  # in degrees
    mass = 84.1  #in kg
    mass_leg = mass/7 # in kg
    g = 9.81  # in m/s
    weight = mass*g

    weight_leg = mass_leg*g   #in Newtons
    d_adductor_attach_hip = 0.05 # in meters
    d_hip_ankle_x = (joint[0] - ground_contact[0])/5  #pixels converted to meters (rough approximation)
    d_hip_knee_x = (joint[0] - mid_joint[0])/5   
    d_hip_oa_x = (joint[0] - opp_ground_contact[0])/5

    #Text set up

    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (00, 185) 
    
    # fontScale 
    fontScale = 3
    
    # Line thickness of 2 px 
    thickness = 2


    #check for 2 leg vs 1 leg condition:
    if (ground_contact[1] - opp_ground_contact[1]) > 10:
        force_normal = weight  # in Newtons        
        adductor_force_y = (force_normal*d_hip_ankle_x - weight_leg*d_hip_knee_x)/d_adductor_attach_hip
        adductor_force_x = adductor_force_y/(math.tan(adductor_angle))

        const = 1000
        hip_force_y = (adductor_force_y - weight_leg + force_normal)
        hip_force_x = -adductor_force_x
        hip_force = math.sqrt(hip_force_x**2 + hip_force_y**2)*35 #hip joint reaction force magnitude
        rf_direction = atan(hip_force_y/hip_force_x)
        start_pt = (joint[0], joint[1])
        end_pt_x = start_pt[0] + hip_force_x//1000*1.5
        end_pt_y = start_pt[1] - hip_force_y//1000*1.5
        
    
        end_pt = (int(end_pt_x), int(end_pt_y))
            
        #cv2.putText(img, "one leg", org, font, fontScale, color, thickness, cv2.LINE_AA)  
    else:
        force_normal = weight/2 # in Newtons   
        d_hip_com = 0.12  # in meters  

        adductor_force_y = (weight*d_hip_com - force_normal*d_hip_oa_x)/d_adductor_attach_hip
        adductor_force_x = adductor_force_y*(math.tan(adductor_angle))      
        const = 1000
        hip_force_y = (weight + adductor_force_y - force_normal)
        hip_force_x = -adductor_force_x
        hip_force = math.sqrt(hip_force_x**2 + hip_force_y**2)*5  #hip joint reaction force magnitude
        rf_direction = atan(hip_force_y/hip_force_x)    
        start_pt = (joint[0], joint[1])
        end_pt_x = start_pt[0] - hip_force_x//1000//10
        end_pt_y = start_pt[1] + hip_force_y//1000//10
        start_pt = (joint[0], joint[1])
        
        #cv2.putText(img, "2 legs", org, font, fontScale, color, thickness, cv2.LINE_AA)
    
        end_pt = (int(end_pt_x), int(end_pt_y))

    #draw force vector
    
    cv2.arrowedLine(img, start_pt, end_pt, color, thickness=thickness)   

    return hip_force


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

        #Left Ankle
        left_ankle = pts2d['detectron2']['custom'][0][0][15]

        #Right Knee
        right_knee = pts2d['detectron2']['custom'][0][0][14]
    
        #Right shoulder
        right_shoulder = pts2d['detectron2']['custom'][0][0][6]
                
        #Right elbow
        right_elbow = pts2d['detectron2']['custom'][0][0][8]
              
        #Right hip
        right_hip = pts2d['detectron2']['custom'][0][0][12]

        #Right Ankle

        right_ankle = pts2d['detectron2']['custom'][0][0][16]               

        shoulder_angle = draw_joint_angles(img, right_shoulder, right_elbow, right_hip)
        hip_force = draw_joint_reaction_force(img, right_hip, right_ankle, right_knee, left_ankle, color = (0, 255, 0), thickness = 5)
        return hip_force, shoulder_angle

      


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

    frame_joint2d = keypoints

    #print(frame_joint2d)

    hip_force, shoulder_angle = draw_body_joints_2d(frame, frame_joint2d, bones=body_edges_17, draw_indices=draw_joint_indices) 
    
    return hip_force, shoulder_angle


