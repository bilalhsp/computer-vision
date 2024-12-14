# Computer Vision - ECE661 - Purdue Univerity

### Homework-1
- **Homogeneous Coordinate Representations**  
  - Representation of points, lines, and conics.
- **Intersection of Lines**  
  - Computing the intersection point using the cross product.
- **Laser Aim Game Simulation**  
  - Utilizing homogeneous coordinates to determine whether the aim is correct (indicated by a green color).

<!-- <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 0px; text-align: center;">
    <img src="./images/hw1/image1.jpg" alt="Original Frame" style="height:150px;">
  </figure>
  <figure style="margin: 0px 0px; text-align: center;">
    <img src="./images/hw1/image2.jpg" alt="Input Image" style="height:150px;">
  </figure>
  <figure style="margin: 0px 0px; text-align: center;">
    <img src="./images/hw1/image3.jpg" alt="Projected Image" style="height:150px;">
  </figure>
    <figure style="margin: 0px 0px; text-align: center;">
    <img src="./images/hw1/image4.jpg" alt="Projected Image" style="height:150px;">
  </figure>
</div>
<p align="center"><strong>Fig. 1:</strong> Laser Aim Simulation</p> -->

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


<!-- <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 20px; text-align: center;">
    <img src="./images/hw2/img1.png" alt="Original Frame" style="width:140px;">
    <figcaption>Original Frame</figcaption>
  </figure>
  <figure style="margin: 40px 20px; text-align: center;">
    <img src="./images/hw2/alex_honnold.jpg" alt="Input Image" style="width:140px;">
    <figcaption>Input Image</figcaption>
  </figure>
  <figure style="margin: 0px 20px; text-align: center;">
    <img src="./images/hw2/projected-image-1a.jpg" alt="Projected Image" style="width:140px;">
    <figcaption>Projected Image</figcaption>
  </figure>
</div>
<p align="center"><strong>Fig. 2:</strong> Image Projected Onto Frame Using Homography</p> -->


<p align="center">
    <img src="./images/hw2/img1.png" alt="Original Frame" style="width:150px; ">
    <img src="./images/hw2/alex_honnold.jpg" alt="Input Image" style="width:150px; ">
    <img src="./images/hw2/projected-image-1a.jpg" alt="Projected Image" style="width:150px;">
</p>
<p align="center"><strong>Fig. 2:</strong> Image Projected Onto Frame Using Homography</p>



<div align="center">
<table style="border:none; display:inline; width: 40%; text-align: center;"> 
    <tr>
    <td style="padding: -10px;"><img src="./images/hw2/img1.png" alt="Original Frame" style="width:150px;"><br>Original Frame</td>
    <td style="padding: -10px;"><img src="./images/hw2/alex_honnold.jpg" alt="Input Image" style="width:150px;"><br>Input Image</td>
    <td style="padding: -10px;"><img src="./images/hw2/projected-image-1a.jpg" alt="Projected Image" style="width:150px;"><br>Projected Image</td>
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


<p align="center">
    <img src="./images/hw3/board_1.png" alt="Distorted Image" style="height:180px;">
    <img src="./images/hw3/board_1-point-to-point.png" alt="Point Matching" style="height:180px;">
    <img src="./images/hw3/board_1-two-step-proj.png" alt="Two-Step" style="height:180px;">
    <img src="./images/hw3/board_1-one-step.png" alt="One-Step" style="height:180px;">
</p>
<p align="center"><strong>Fig. 3:</strong> Metric Rectification Example</p>



<!-- <p align="center">
    <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1.png" alt="Distorted Image" style="height:150px;">
    <figcaption>Distorted Image</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-point-to-point.png" alt="Point Matching" style="height:150px;">
    <figcaption>Point Matching</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-two-step-proj.png" alt="Two-Step" style="height:150px;">
    <figcaption>Two-Step</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-one-step.png" alt="One-Step" style="height:150px;">
    <figcaption>One-Step</figcaption>
  </figure>
</p>
<p align="center"><strong>Fig. 3:</strong> Metric Rectification Example</p> -->

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1.png" alt="Distorted Image" style="height:150px;">
    <figcaption>Distorted Image</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-point-to-point.png" alt="Point Matching" style="height:150px;">
    <figcaption>Point Matching</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-two-step-proj.png" alt="Two-Step" style="height:150px;">
    <figcaption>Two-Step</figcaption>
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw3/board_1-one-step.png" alt="One-Step" style="height:150px;">
    <figcaption>One-Step</figcaption>
  </figure>
</div>
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

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/harris_kps_sigma_12_hovde_3.png" alt="Hovde_3" style="height:150px;">
    <!-- <figcaption>Hovde</figcaption> -->
  </figure>
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/harris_kps_sigma_12_temple_2.png" alt="Temple" style="height:150px;">
    <!-- <figcaption>Rectified</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 4:</strong> Harris Corner Detection</p>



<div style="display: flex; justify-content: center; align-items: flex-start;">
  <!-- <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/harris_kps_sigma_12_hovde_3.png" alt="Hovde_3" style="height:200px;">
    <figcaption>Hovde</figcaption>
  </figure> -->
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw4/SIFT_temple_.png" alt="Temple" style="height:150px;">
    <!-- <figcaption>Rectified</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 5:</strong> SIFT Points Matching</p>

[View Full Report](./reports/hw4_BilalAhmed.pdf)

### Homework-5
- **Image Mosaicing**  
  - Automated homography estimation between image pairs.  
  - Detected and matched interest points.  
  - Rejected outliers using RANSAC.  
  - Refined homography using Levenberg-Marquardt (LM).  
  - Created a panoramic view using homographies.

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 2px; text-align: center;">
    <img src="./images/hw5/1.jpg" alt="Image 1" style="width:120px;">
    <!-- <figcaption>Image 1</figcaption> -->
  </figure>
  <figure style="margin: 0px 2px; text-align: center;">
    <img src="./images/hw5/2.jpg" alt="Image 2" style="width:120px;">
    <!-- <figcaption>Image 2</figcaption> -->
  </figure>
  <figure style="margin: 0px 2px; text-align: center;">
    <img src="./images/hw5/3.jpg" alt="Image 3" style="width:120px;">
    <!-- <figcaption>Image 3</figcaption> -->
  </figure>
  <figure style="margin: 0px 2px; text-align: center;">
    <img src="./images/hw5/4.jpg" alt="Image 4" style="width:120px;">
    <!-- <figcaption>Image 4</figcaption> -->
  </figure>
  <figure style="margin: 0px 2px; text-align: center;">
    <img src="./images/hw5/5.jpg" alt="Image 5" style="width:120px;">
    <!-- <figcaption>Image 5</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 6:</strong> Overlapping Images of the Fountain</p>

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 10px; text-align: center;">
    <img src="./images/hw5/mosaic_image_task1_using_lm.png" alt="Image Mosaic" style="height:150px;">
    <!-- <figcaption>Image Mosaic</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 7:</strong> Image Mosaic Created Using Overlapping Images</p>

[View Full Report](./reports/hw5_BilalAhmed.pdf)

### Homework-6  
- **Image Segmentation**  
  - Otsu's Algorithm:  
    - Segmentation using RGB channels.  
    - Texture-based segmentation.  
  - Contour Extraction:  
    - Applied morphological operations for contour extraction.


<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0px 5px; text-align: center;">
    <img src="./images/hw6/flower_small.png" alt="Flower Image" style="width:170px;">
    <figcaption>Original Flower Image</figcaption>
  </figure>
  <figure style="margin: 0px 5px; text-align: center;">
    <img src="./images/hw6/flower_small-mask-combined.jpg" alt="Segmented Binary Mask" style="width:170px;">
    <figcaption>Segmented Binary Mask</figcaption>
  </figure>
  <figure style="margin: 0px 5px; text-align: center;">
    <img src="./images/hw6/flower_small-contour-map.jpg" alt="Contour Map" style="width:170px;">
    <figcaption>Extracted Contour Map</figcaption>
  </figure>
</div>
<p align="center"><strong>Fig. 8:</strong> Segmentation and Contour Extraction Example</p>

[View Full Report](./reports/hw6_BilalAhmed.pdf)

### Homework-7  
- **Weather Classification Using Various Features**  
    - Local Binary Patterns (LBP): Utilized for texture-based feature extraction.  
    - Deep Neural Network Features: Extracted high-level features using:  
        + VGG  
        + ResNet  
    - Adaptive Instance Normalization (AdaIN): Leveraged style-based features for classification.  


<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 10px 5px; text-align: center;">
    <img src="./images/hw7/AdaIN_c_matrix.png" alt="Confusion Matrix" style="width:180px;">
    <!-- <figcaption>Confusion Matrix</figcaption> -->
  </figure>
  <figure style="margin: -10px 5px; text-align: center;">
    <img src="./images/hw7/AdaIN_example_correct.png" alt="Correct Classification Example" style="width:180px;">
    <!-- <figcaption>Correct Classification Example</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 9:</strong> Confusion Matrix and Classification Example</p>

[View Full Report](./reports/hw7_BilalAhmed.pdf)

### Homework-8  
- **Camera Calibration**  
  - Implemented Zhang's Algorithm:  
    - Detecting corners in a calibration pattern.  
    - Calculating intrinsic parameters.  
    - Calculating extrinsic parameters.  
    - Refining parameter estimates using the Levenberg-Marquardt (LM) algorithm.

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 15px 5px; text-align: center;">
    <img src="./images/hw8/Pic_5.jpg" alt="Calibration Pattern" style="width:140px;">
    <figcaption style="margin-top: 12px;">Calibration Pattern View</figcaption>
  </figure>
  <figure style="margin: 0px 5px; text-align: center;">
    <img src="./images/hw8/edges-Pic_5.jpg" alt="Detected Boxes" style="width:180px;">
    <figcaption>Detected Boxes</figcaption>
  </figure>
  <figure style="margin: 0px 5px; text-align: center;">
    <img src="./images/hw8/corners-Pic_5.jpg" alt="Detected Corners" style="width:180px;">
    <figcaption>Detected Corners</figcaption>
  </figure>
</div>
<p align="center"><strong>Fig. 10:</strong> Calibration Pattern</p>


<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 0 10px; text-align: center;">
    <img src="./images/hw8/dataset1-camera-poses.png" alt="Recreated Camera Poses" style="height:200px;">
    <!-- <figcaption>Recreated Camera Poses</figcaption> -->
  </figure>
</div>
<p align="center"><strong>Fig. 11:</strong> Recreated Camera Poses</p>

[View Full Report](./reports/hw8_BilalAhmed.pdf)


### Homework-9

- **Projective Stereo Reconstruction**:
    - Stereo Rectification: Aligning images to simplify correspondence matching by making epipolar lines horizontal.
    - Interest Point Detection: Identifying distinct features across images for matching.
    - Projective Reconstruction: Reconstructing 3D points from stereo correspondences using projective geometry.

- **Dense Stereo Matching**: Generating disparity maps to find pixel-wise correspondences between stereo images.

- **Extraction of Dense Correspondences using Depth Maps**: Refining disparity maps to generate dense depth information for accurate 3D reconstruction.



<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 10px; text-align: center;">
    <img src="./images/hw9/image-keypoints-projected-world-3d-view-2.jpg" alt="Projective Stereo Reconstruction 3D View" style="height:200px;">
    <figcaption style="margin-top: 12px;">Projective Stereo Reconstruction 3D View</figcaption>
  </figure>
</div>
<p align="center"><strong>Fig. 12:</strong> Projective Stereo Reconstruction</p>


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


<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 10px; text-align: center;">
    <img src="./images/hw10/accuracy.png" alt="Face Recognition Accuracy" style="height:250px;">
    <figcaption style="margin-top: 5px; font-size: 0.9em; color: #555;">Fig. 13: Face Recognition Accuracy</figcaption>
  </figure>
</div>


<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="margin: 10px 5px; text-align: center;">
    <img src="./images/hw10/training_fpr.png" alt="Training False Positive Rate" style="width:250px;">
    <figcaption style="margin-top: 10px; font-size: 0.9em; color: #555;">Training</figcaption>
  </figure>
  <figure style="margin: 10px 5px; text-align: center;">
    <img src="./images/hw10/testing_fpr_fnr.png" alt="Testing False Positive and Negative Rates" style="width:250px;">
    <figcaption style="margin-top: 10px; font-size: 0.9em; color: #555;">Testing</figcaption>
  </figure>
</div>
<p align="center"><strong>Fig. 14:</strong> Object Detection: False Positive and False Negative Rates</p>

[View Full Report](./reports/hw10_BilalAhmed.pdf)