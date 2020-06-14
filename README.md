# RealTimeJointAnalysis
Motivated by personal injuries in the past and a natural curiosity about human motion, this project looks to highlight the use of computer vision to aid in rehabilitation for injured patients. From personal experience, I know it can be difficult to assess progress when going through recovery. I believe that computer vision systems can be developed to help doctors in diagnosing patients as well as providing valuable biofeedback to patients. 

Using a live webcam feed, the Joint Ananlyzer does 3 things:  
1. Detects joints and displays them along with bones on top of each frame  
2. Calculates a rough approximation of the joint reaction force acting at the hip  
3. Dispays joint angle and hip force information for the user

# Background Information

* Injuries that limit body mobility can cause depression or anxiety [1] and can be an economic burden [2]  
* Some promise has been shown in experiments researching visual biofeedback [3]  
* Inspired by the work from Facebook Research Group: led by Dario Pavllo -- [4]  
* Updated the project to work with Detectron2 with help from: darkAlert -- [5]  
* Added functionality to work with live webcam feed  
* Added joint angle measurement at the shoulder joint (relative to torso)  
* Calculated and displayed joint reaction force at the hip based on 2 biomechanical cases: 1 legged vs 2 legged stance  

# How to Implement

* Make sure you have a webcam connected to your computer  
* Install the requirements as described by the install section of [4]  
* Install detectron2:  
* Clone this repository  
* Run the pose_analyzer_webcam.py script

&nbsp;

# Results

&nbsp;

<div align="center">

**Real Time Joint Detection**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/joint_detection_001.png" width="360">
</p>

&nbsp;

<div align="center">

**Single Leg Biomechanics**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/one_leg_hip_force.png" width="520">
</p>

&nbsp;

<div align="center">

**Double leg Biomechanics**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/two_leg_hip_force.png" width="520">
</p>

&nbsp;

<div align="center">

**Offline Joint Detection**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/wave_squat.gif" width="520">
</p>

&nbsp;

<div align="center">

**Real Time Joint Velocity using MATLAB**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/RT_JointAnalysis.gif" width="520">
</p>

&nbsp;
&nbsp;

Pose classification and joint angle measurement has the potential to be used in many applications including sports and rehabiliation performance as well as human-computer interaction systems

# Current Limitations

* Low frame rate (~4fps): More powerful gpu could provide higher frame rates
* Simple biomechanics model: Currently accounts for only one (adductor) muscle at the hip
* Only works for single person in frame: If multiple people are in frame, only one person will be recognized

# Future Direction
* Implement 3D joint estimation: Transfer this project into a 3D (more realistic) analysis of forces acting on joints
* Increase biomechanical model complexity: Include a more complex analysis of joint forces by including more muscles
* Incorporate pose detection: Utilize a pose detector to display the relevant joint angles for each specific pose

# References
[1] Shafrin J, Sullivan J, Goldman DP, Gill TM (2017) The association between observed mobility and quality of life in the near elderly  
[2] Goldman DP et al. (2018) Long-Term Health and Economic Value of Improved Mobility among Older Adults in the United States  
[3] Barandasa M, Gamboab H, Fonseca J (2015) A RealTime Biofeedback System Using Visual User Interface for Physical Rehab  
[4] https://github.com/facebookresearch/VideoPose3D  
[5] https://github.com/darkAlert/VideoPose3d_with_Detectron2
