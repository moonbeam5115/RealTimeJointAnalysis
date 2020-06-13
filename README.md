# RealTimeJointAnalysis
While training to increase my athletic performance, I noticed the importance of mobility in decreasing my chance of injury  while performing athletic movements. The ability of our muscles to lengthen and contract rapidly is necessary in order to execute athletic movements such as sprinting and jumping. If our muscles are not accustomed to stretching beyond a certain extent, muscle pulls, strains, or even tears may occur, leading to injuries that force athletes to cease activity. 

My idea with this project is to provide a method for joint angle measurement, which could be used to assess an athlete's mobility. This information could be incorporated into their training program, so as to not include movements or exercises that may cause injury to the athlete. It could also be used to determine which athlete could benefit more from mobility and flexibility routines. Finally, in a rehabilitation setting, the measure of joint angles could be used to measure progress for rehabilitation patients as they perform neuromotor exercises to regain range of motion.

Given an image, the Pose Analyzer does 3 things:
1. Classifies the input image's pose: squatting, bending, or raising arms
2. Overlays the predicted human joint position for the input image
3. Draw and display joint angle information currently only works for specific instances of squatting poses. More work will need to be done to generalize the joint angle measurement for all poses and cases

# Data Sources

* Took 291 personal pictures of the following poses: squatting, bending, and raising arms
* Took 99 pictures from google search for the following poses: squatting, bending, and raising arms

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
