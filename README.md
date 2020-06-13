# RealTimeJointAnalysis
Motivated by personal injuries in the past and a natural curiosity about human motion, this project looks to highlight the use of computer vision to aid in rehabilitation for injured patients. From personal experience, I know it can be difficult to assess progress when going through recovery. I believe that computer vision systems can be developed to help doctors in diagnosing patients as well as providing valuable biofeedback to patients. 

Using a live webcam feed, the Joint Ananlyzer does 3 things:  
1. Detects joints and displays them along with bones on top of each frame  
2. Calculates a rough approximation of the joint reaction force acting at the hip  
3. Dispays joint angle and hip force information for the user

# Background Information

* This project utilizes 2 deep learning models to classify and then detect joint positions for a given input image
* Pose clasification is a much easier problem than pose estimation
* This project utilizes the VGG16 model as a base model for transfer learning and predicting poses from images
* Pose estimation is difficult due to joint occlusion, clothing, the degrees of freedom in a human body and more
* This project utilizes the OpenPose model by the folks over at CMU to detect joint positions. Their method for joint detection involves part affinity maps (PAFs) and heat maps -- github link
* This project also aims to measure key joint angles depending on the detected pose -- (to be implemented at a future time)


&nbsp;

# Results

The predictive results of both the classifier and joint detection models were very impressive. Despite being only trained on 285 images, the pose classifier performed at ~97% accuracy for the validation and test sets. Although very impressive, the models did make mistakes that humans would consider "silly." This begs the question as to how these algorithms are actually "learning" and what it means to learn something altogether.

&nbsp;


<div align="center">

**Pose Classification, Joint Detection and Joint Angle Measurement**
</div>

<p align="center">
<img src="https://github.com/moonbeam5115/RealTimeJointAnalysis/blob/master/imgs/joint_detection_001.png" width="360">
</p>

&nbsp;
&nbsp;

# Conlusion

* Transfer learning can be a very effective way to train a classifier given smaller datasets
* OpenPose provides very accurate human pose estimation and can be used for joint angle measurement
* Although very impressive, these deep learning models still make errors on seemingly simple classification problems
* Pose classification and joint angle measurement has the potential to be used in many applications including sports and rehabiliation performance as well as human-computer interaction systems

# Future Direction
*

* 

* 

# References
*description
