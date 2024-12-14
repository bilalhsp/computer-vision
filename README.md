# Computer Vision - ECE661 - Purdue Univerity

### Homework-1
- **Homogeneous Coordinate Representations**  
  - Representation of points, lines, and conics.
- **Intersection of Lines**  
  - Computing the intersection point using the cross product.
- **Laser Aim Game Simulation**  
  - Utilizing homogeneous coordinates to determine whether the aim is correct (indicated by a green color).



<p align="center">
    <img src="./images/hw1/image1.jpg" alt="Original Frame" style="height:120px; margin-right: 10px;">
    <img src="./images/hw1/image2.jpg" alt="Input Image" style="height:120px; margin-right: 10px;">
    <img src="./images/hw1/image3.jpg" alt="Projected Image" style="height:120px; margin-right: 10px;">
    <img src="./images/hw1/image4.jpg" alt="Projected Image" style="height:120px;">
</p>
<p align="center"><strong>Fig. 1:</strong> Laser Aim Simulation</p>




<!-- <p align="center"><strong></strong> Common caption for both images</p> -->

[View Full Report](./reports/hw1_BilalAhmed.pdf)

### Homework-2
- **Introduction to Homography**  
  - Understanding the concept and mathematical formulation of homography.
- **Using Homography for Image Projection**  
  - Projecting an image onto a frame in another image using homography.




<!-- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;
Original Frame &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Input Image&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Projected Image -->




<p align="center" style="font-size: 0.9em; color: #555;">
<strong>Original Frame</strong> 
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
<strong>Input Image</strong>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp; &nbsp;&nbsp;
<strong>Projected Image</strong>
</p>

<div align="center">
<table style="border:none; display:inline; width: 40%; text-align: center;"> 
    <tr>
    <td style="padding: -10px;"><img src="./images/hw2/img1.png" alt="Original Frame" style="width:150px;"></td>
    <td style="padding: -10px;"><img src="./images/hw2/alex_honnold.jpg" alt="Input Image" style="width:150px;"></td>
    <td style="padding: -10px;"><img src="./images/hw2/projected-image-1a.jpg" alt="Projected Image" style="width:150px;"></td>
  </tr>
</table>
</div>
<p align="center"><strong>Fig. 2:</strong> Image Projected Onto Frame Using Homography</p>



[View Full Report](./reports/hw2_BilalAhmed.pdf)


### Homework-3
- **Metric Rectification**  
  - Correcting projective and affine distortions in images.
- **Point-to-Point Rectification**  
  - Using corresponding points between the distorted image and the physical or undistorted view.
- **Two-step Rectification**  
  - Removing projective distortion using a vanishing line.
  - Eliminating affine distortion using orthogonal lines.
- **One-step Rectification**  
  - Applying a transformation of a degenerate conic to correct both projective and affine distortions.



<!-- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Distorted Image&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Point Matching&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Two-Step&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
<!-- One-Step -->

<p align="center" style="font-size: 0.9em; color: #555;">
<strong>Distorted Image</strong> 
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;
<strong>Point Matching</strong>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<strong>Two-Step</strong>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<strong>One-Step</strong>
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;
</p>


<p align="center">
    <img src="./images/hw3/board_1.png" alt="Distorted Image" style="height:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw3/board_1-point-to-point.png" alt="Point Matching" style="height:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw3/board_1-two-step-proj.png" alt="Two-Step" style="height:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw3/board_1-one-step.png" alt="One-Step" style="height:180px;">
</p>
<p align="center"><strong>Fig. 3:</strong> Metric Rectification Example</p>


[View Full Report](./reports/hw3_BilalAhmed.pdf)

### Homework-4
- **Harris Corner Detection**  
  - Implemented interest point detection.  
  - Matched corresponding points between pairs of images using:  
    + Sum of Squared Differences (SSD).  
    + Normalized Cross-Correlation (NCC).  
- **Alternative Methods for Interest Point Detection and Matching**  
  - Used Scale-Invariant Feature Transform (SIFT).  
  - Applied deep learning models, SuperPoint and SuperGlue, for enhanced point detection and matching.

<!-- <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/harris_kps_sigma_12_hovde_3.png" alt="Hovde_3" style="height:150px;">
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/harris_kps_sigma_12_temple_2.png" alt="Temple" style="height:150px;">
  </figure>
</div>
<p align="center"><strong>Fig. 4:</strong> Harris Corner Detection</p> -->

<p align="center">
    <img src="./images/hw4/harris_kps_sigma_12_hovde_3.png" alt="Hovde_3" style="height:150px;">
    &nbsp;&nbsp;
    <img src="./images/hw4/harris_kps_sigma_12_temple_2.png" alt="Temple" style="height:150px;">
</p>
<p align="center"><strong>Fig. 4:</strong> Harris Corner Detection</p>



<p align="center">
    <img src="./images/hw4/SIFT_temple_.png" alt="Temple" style="height:150px;">
</p>
<p align="center"><strong>Fig. 5:</strong> SIFT Points Matching</p>

[View Full Report](./reports/hw4_BilalAhmed.pdf)

### Homework-5
- **Image Mosaicing**  
  - Automated homography estimation between image pairs.  
  - Detected and matched interest points.  
  - Rejected outliers using RANSAC.  
  - Refined homography using Levenberg-Marquardt (LM).  
  - Created a panoramic view using homographies.

<p align="center">
    <img src="./images/hw5/1.jpg" alt="Image 1" style="width:120px;">
    &nbsp;&nbsp;
    <img src="./images/hw5/2.jpg" alt="Image 2" style="width:120px;">
    &nbsp;&nbsp;
    <img src="./images/hw5/3.jpg" alt="Image 3" style="width:120px;">
    &nbsp;&nbsp;
    <img src="./images/hw5/4.jpg" alt="Image 4" style="width:120px;">
    &nbsp;&nbsp;
    <img src="./images/hw5/5.jpg" alt="Image 5" style="width:120px;">
</p>

<p align="center"><strong>Fig. 6:</strong> Overlapping Images of the Fountain</p>

<p align="center">
    <img src="./images/hw5/mosaic_image_task1_using_lm.png" alt="Image Mosaic" style="height:150px;">
</p>
<p align="center"><strong>Fig. 7:</strong> Image Mosaic Created Using Overlapping Images</p>

[View Full Report](./reports/hw5_BilalAhmed.pdf)

### Homework-6  
- **Image Segmentation**  
  - Otsu's Algorithm:  
    - Segmentation using RGB channels.  
    - Texture-based segmentation.  
  - Contour Extraction:  
    - Applied morphological operations for contour extraction.


<p align="center">
    <img src="./images/hw6/flower_small.png" alt="Flower Image" style="width:170px;">
    &nbsp;&nbsp;
    <img src="./images/hw6/flower_small-mask-combined.jpg" alt="Segmented Binary Mask" style="width:170px;">
    &nbsp;&nbsp;
    <img src="./images/hw6/flower_small-contour-map.jpg" alt="Contour Map" style="width:170px;">
</p>

<p align="center"><strong>Fig. 8:</strong> Segmentation and Contour Extraction Example</p>

[View Full Report](./reports/hw6_BilalAhmed.pdf)

### Homework-7  
- **Weather Classification Using Various Features**  
    - Local Binary Patterns (LBP): Utilized for texture-based feature extraction.  
    - Deep Neural Network Features: Extracted high-level features using:  
        + VGG  
        + ResNet  
    - Adaptive Instance Normalization (AdaIN): Leveraged style-based features for classification.  


<p align="center">
    <img src="./images/hw7/AdaIN_c_matrix.png" alt="Confusion Matrix" style="width:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw7/AdaIN_example_correct.png" alt="Correct Classification Example" style="width:180px;">
</p>

<p align="center"><strong>Fig. 9:</strong> Confusion Matrix and Classification Example</p>

[View Full Report](./reports/hw7_BilalAhmed.pdf)

### Homework-8  
- **Camera Calibration**  
  - Implemented Zhang's Algorithm:  
    - Detecting corners in a calibration pattern.  
    - Calculating intrinsic parameters.  
    - Calculating extrinsic parameters.  
    - Refining parameter estimates using the Levenberg-Marquardt (LM) algorithm.

<p align="center">
    <img src="./images/hw8/Pic_5.jpg" alt="Calibration Pattern" style="width:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw8/edges-Pic_5.jpg" alt="Detected Boxes" style="width:180px;">
    &nbsp;&nbsp;
    <img src="./images/hw8/corners-Pic_5.jpg" alt="Detected Corners" style="width:180px;">
</p>

<p align="center"><strong>Fig. 10:</strong> Calibration Pattern</p>


<p align="center">
    <img src="./images/hw8/dataset1-camera-poses.png" alt="Recreated Camera Poses" style="height:200px;">
</p>

<p align="center"><strong>Fig. 11:</strong> Recreated Camera Poses</p>

[View Full Report](./reports/hw8_BilalAhmed.pdf)


### Homework-9

- **Projective Stereo Reconstruction**:
    - Stereo Rectification: Aligning images to simplify correspondence matching by making epipolar lines horizontal.
    - Interest Point Detection: Identifying distinct features across images for matching.
    - Projective Reconstruction: Reconstructing 3D points from stereo correspondences using projective geometry.

- **Dense Stereo Matching**: Generating disparity maps to find pixel-wise correspondences between stereo images.

- **Extraction of Dense Correspondences using Depth Maps**: Refining disparity maps to generate dense depth information for accurate 3D reconstruction.



<p align="center">
    <img src="./images/hw9/image-keypoints-projected-world-3d-view-2.jpg" alt="Projective Stereo Reconstruction 3D View" style="height:200px;">
</p>
<p align="center"><strong>Fig. 12:</strong> Projective Stereo Reconstruction 3D View</p>


[View Full Report](./reports/hw9_BilalAhmed.pdf)

### Homework-10

- **Face Recognition**
    - Dimensionality Reduction:
        - Principal Component Analysis (PCA)
        - Linear Discriminant Analysis (LDA)
        - Variational Autoencoder (VAE)
    - Classification:
        - Nearest Neighbor Classifier

- **Object Detection**
    - Cascaded AdaBoost Classifiers


<p align="center">
    <img src="./images/hw10/accuracy.png" alt="Face Recognition Accuracy" style="height:250px;">
</p>
<p align="center"> <strong>Fig. 13:</strong> Face Recognition Accuracy</p>


<p align="center" style="font-size: 0.9em; color: #555;"><strong>Training</strong> 
&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;  
<strong>Testing</strong></p>

<p align="center">
    <img src="./images/hw10/training_fpr.png" alt="Training False Positive Rate" style="width:250px;">
    &nbsp;&nbsp;
    <img src="./images/hw10/testing_fpr_fnr.png" alt="Testing False Positive and Negative Rates" style="width:250px;">
</p>
<p align="center"><strong>Fig. 14:</strong> Object Detection: False Positive and False Negative Rates</p>

[View Full Report](./reports/hw10_BilalAhmed.pdf)