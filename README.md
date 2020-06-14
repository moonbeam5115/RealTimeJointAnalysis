# RealTimeJointAnalysis
Motivated by personal injuries in the past and a natural curiosity about human motion, this project looks to highlight the use of computer vision to aid in rehabilitation for injured patients. From personal experience, I know it can be difficult to assess progress when going through recovery. I believe that computer vision systems can be developed to help doctors in diagnosing patients as well as providing valuable biofeedback to patients. 

Using a live webcam feed, the Joint Ananlyzer does 3 things:  
1. Detects joints and displays them along with bones on top of each frame  
2. Calculates a rough approximation of the joint reaction force acting at the hip  
3. Dispays joint angle and hip force information for the user

# Background Information

* Injuries that limit body mobility can cause depression or anxiety[1] and can be an economic burden [2]  
* Some promise has been shown in experiments researching visual biofeedback [3]  
* Inspired by the work from Facebook Research Group: led by Dario Pavllo -- github link [4]  
* Updated the project to work with Detectron2 with help from: -- github link [5]  
* Added functionality to work with live webcam feed  
* Added joint angle measurement at the shoulder joint (relative to torso)  
* Calculated and displayed joint reaction force at the hip based on 2 biomechanical cases: 1 legged vs 2 legged stance  


&nbsp;

# Results



&nbsp;


<div align="center">

**Pose Classification, Joint Detection and Joint Angle Measurement**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/joint_detection_001.png" width="360">
</p>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/tree/master/imgs/one_leg_hip_force.png" width="360">
</p>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/tree/master/imgs/two_legs_hip_force.png" width="360">
</p>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/tree/master/imgs/wave_squat.gif" width="360">
</p>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/tree/master/imgs/RT_JointAnalysis.gif" width="360">
</p>

&nbsp;
&nbsp;

# Limitations

* 
* 
* 
* Pose classification and joint angle measurement has the potential to be used in many applications including sports and rehabiliation performance as well as human-computer interaction systems

# Future Direction
*

* 

* 

# References
[1]  
[2]  
[3]  
[4]  
[5]
