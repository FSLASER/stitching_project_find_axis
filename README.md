# stitching_project_find_axis

## Description

This project is designed to stitch together a series of images and find the corrected axis of rotation by detecting feature points and adjusting for alignment. Below is a step-by-step description of the workflow, from feature point detection to user interaction and final image stitching.

### 1. Feature Points Detection
- **SIFT (Scale-Invariant Feature Transform):**
  - The workflow begins by loading pairs of consecutive images from the specified directory.
  - For each image pair, the SIFT algorithm is used to detect and compute keypoints and their corresponding descriptors. These keypoints represent distinct and identifiable features within the images (such as corners, edges, etc.).

### 2. Feature Points Matching
- **Brute Force Matcher (BFMatcher):**
- **RANSAC (Random Sample Consensus):**

### 3. Calculate Displacements
- **Displacement Calculation:**
  - For each pair of matched inlier keypoints, the displacement between the corresponding points in the two images is calculated.
  - The average vertical displacement (`y-shift`) is computed, which represents the relative shift between the images. This value is crucial for aligning the images during the stitching process.

### 4. Image Recenter and Rotation
- **Recenter and Rotate:**
  - Once the displacements have been calculated, the images are recentered and rotated based on the keypoint displacements.
  - The script allows for the adjustment of axis shifts (both left and right) to fine-tune the alignment of each image. This is done using a function that computes a rotation matrix to align the images horizontally based on the detected displacements.
  - This recentering and rotation step ensures that the keypoints align along a common horizontal axis, which is necessary for accurate stitching.

### 5. User Interaction with Trackbars
- **Interactive Trackbars:**
  - The code provides a graphical interface where users can interactively adjust parameters using trackbars:
    - **Global Shift:** Allows the user to fine-tune the overall vertical shift applied during stitching.
    - **Axis Shift Left and Right:** Enables adjustments to the left and right axis shifts, which are particularly useful for correcting any skew in the images.
  - As the user adjusts these parameters, the images are dynamically recentered and re-stitched, providing immediate visual feedback on the effect of the changes.

### 6. Image Stitching
- **Stitching Process:**
  - After the images have been recentered and aligned, they are stitched together into a single image.
  - The stitching process involves vertically aligning the images based on the computed displacements and the user-adjusted parameters.
  - The script ensures a seamless transition between the images, taking into account the calculated shifts and any adjustments made by the user.

- **Saving the Final Image:**
  - Once the user is satisfied with the alignment and stitching, the final stitched image is displayed.
  - The image is then saved to the local directory as `final_stitched_image.jpg`, preserving the adjustments made during the process.

### Summary

This workflow outlines how the code systematically processes a series of images to detect keypoints, match features, calculate displacements, recenter and rotate images, and finally stitch them into a single image. The user can interactively adjust key parameters, such as global shift and axis shifts, to ensure the images are perfectly aligned. The end result is a well-stitched image that combines multiple input images into a cohesive whole.



## Code
```bash

import numpy as np
import cv2
import os
import re


def numerical_sort(value):
    match = re.search(r'(\d+\.?\d*)', value)
    return float(match.group(1)) if match else 0


def calculate_projected_length(diameter, arc_length_mm):
    circumference = np.pi * diameter
    theta = (arc_length_mm / circumference) * 2 * np.pi
    radius = diameter / 2
    projected_length = 2 * radius * np.sin(theta / 2)
    return projected_length


def crop_image(img, x1, x2, y1, y2, axis_shift=0):
    height, width = img.shape[:2]
    y1 = int(y1 + axis_shift)
    y2 = int(y2 + axis_shift)
    return img[max(0, y1):min(height, y2), x1:x2]


def sift_feature_matching(img1, img2, ratio_threshold):
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one or both images.")
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    # Extract matched keypoints
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Apply RANSAC to filter out outliers
    if len(pts1) >= 4:
        _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        inliers = mask.ravel().tolist()
        good_matches = [good_matches[i] for i in range(len(good_matches)) if inliers[i]]

    # Calculate shifts only from inliers
    shifts = []
    for match in good_matches:
        point1 = keypoints1[match.queryIdx].pt
        point2 = keypoints2[match.trainIdx].pt
        shift = (point2[0] - point1[0], point2[1] - point1[1])
        if np.abs(shift[0]) < 5 and shift[1] > 0:
            shifts.append(shift)

    return shifts


def process_directory(directory, x1, x2, y1, y2, ratio_threshold, axis_shift_left=0, axis_shift_right=0):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')],
                         key=numerical_sort)
    total_shifts = []

    for i in range(len(image_files) - 1):
        img1_path = os.path.join(directory, image_files[i])
        img2_path = os.path.join(directory, image_files[i + 1])

        print(f"Processing img1: {img1_path} and img2: {img2_path}")

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"One or both images could not be loaded: {img1_path}, {img2_path}")
            continue

        cropped_img1 = crop_image(img1, x1, x2, y1, y2, axis_shift_left)  # Using axis_shift_left for now
        cropped_img2 = crop_image(img2, x1, x2, y1, y2, axis_shift_right)  # Using axis_shift_right for now

        shifts = sift_feature_matching(cropped_img2, cropped_img1, ratio_threshold)
        total_shifts.append(shifts)

        if shifts:
            avg_shift_y = np.mean([shift[1] for shift in shifts])
            print(f"Average y-shift between {image_files[i]} and {image_files[i + 1]}: {avg_shift_y:.2f} pixels")
        else:
            print(f"No valid shifts found between {image_files[i]} and {image_files[i + 1]}")

    all_shifts = [shift[1] for pair_shifts in total_shifts for shift in pair_shifts if pair_shifts]
    global_avg_shift_y = np.mean(all_shifts) if all_shifts else 0
    print(f"Global average y-shift: {global_avg_shift_y:.2f} pixels")

    return total_shifts, global_avg_shift_y


def recenter_and_rotate(img, x1, y1, x2, y2):
    img_height, img_width = img.shape[:2]
    img_center_x = img_width / 2
    img_center_y = img_height / 2
    obj_center_x = (x1 + x2) / 2
    obj_center_y = (y1 + y2) / 2

    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad)

    M_rot = cv2.getRotationMatrix2D((obj_center_x, obj_center_y), angle_deg, 1.0)

    rotated_img = cv2.warpAffine(img, M_rot, (img_width, img_height))

    rot_center_x = (M_rot[0, 0] * obj_center_x + M_rot[0, 1] * obj_center_y + M_rot[0, 2])
    rot_center_y = (M_rot[1, 0] * obj_center_x + M_rot[1, 1] * obj_center_y + M_rot[1, 2])

    x_trans_after_rot = img_center_x - rot_center_x
    y_trans_after_rot = img_center_y - rot_center_y

    M_trans = np.array([[1, 0, x_trans_after_rot], [0, 1, y_trans_after_rot]])

    transformed_img = cv2.warpAffine(rotated_img, M_trans, (img_width, img_height))

    return transformed_img


def update_stitching(val):
    global stitch_img, last_valid_shift, image_files, directory, num_imgs, num_overlap_imgs

    # Get the global average shift from the trackbar and convert it back to a float
    global_avg_shift = cv2.getTrackbarPos('Shift', 'Stitched Image') / 100.0
    axis_shift_left = cv2.getTrackbarPos('Axis Shift Left', 'Stitched Image') - 50
    axis_shift_right = cv2.getTrackbarPos('Axis Shift Right', 'Stitched Image') - 50

    recentered_images = []
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(directory, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image {img_file}")
            continue

        height, width = img.shape[:2]
        x1, x2 = 0, width - 1  # The far left and right of the image
        y1, y2 = int(height / 2 + axis_shift_left), int(height / 2 + axis_shift_right)

        transformed_img = recenter_and_rotate(img, x1, y1, x2, y2)
        recentered_images.append(transformed_img)

    stitch_img = None
    for i in range(num_imgs):
        stitch_img = stitch(i, global_avg_shift, recentered_images[i], stitch_img, axis_shift_left=axis_shift_left,
                            axis_shift_right=axis_shift_right, num_imgs=num_imgs, num_overlap_imgs=num_overlap_imgs)

    # Display the stitched image
    if stitch_img is not None:
        cv2.imshow('Stitched Image', stitch_img)
        # Save the final stitched image
        cv2.imwrite('final_stitched_image.jpg', stitch_img)
        print("Final stitched image saved as 'final_stitched_image.jpg'")
    else:
        print("No images to display after stitching.")

    print(f"Final adjusted values:")
    print(f"Global average shift: {global_avg_shift:.2f} pixels")
    print(f"Axis shift left: {axis_shift_left:.2f} pixels")
    print(f"Axis shift right: {axis_shift_right:.2f} pixels")


def stitch(img_index, global_avg_shift, recentered_img, stitch_img=None, axis_shift_left=0, axis_shift_right=0,
           num_imgs=0, num_overlap_imgs=0):
    height, width = recentered_img.shape[:2]

    center_line = int(height / 2)

    if img_index == 0:
        stitch_img = recentered_img[0:int(center_line + global_avg_shift / 2), :]
    elif img_index < num_imgs - num_overlap_imgs:
        new_slice = recentered_img[
                    int(center_line - global_avg_shift / 2):int(center_line + global_avg_shift / 2), :]
        stitch_img = np.vstack((stitch_img, new_slice))
    elif img_index == num_imgs - num_overlap_imgs:
        new_slice = recentered_img[int(center_line - global_avg_shift / 2):, :]
        stitch_img = np.vstack((stitch_img, new_slice))

    return stitch_img


import os


def main():
    global stitch_img, last_valid_shift, image_files, directory, num_imgs, num_overlap_imgs
    input_directory = 'ori_rotary_pic'  # Original directory containing the images
    output_directory = 'recentered'
    diameter = 66
    arc_length_mm = 0.5

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    print("Enter the axis of rotation in x1, x2, y1, y2 for recentering the images, or input '0000' if the images are already recentered.")
    recenter_input = input("Enter values (format: x1 x2 y1 y2): ").strip()

    if recenter_input != "0000":
        try:
            x1, x2, y1, y2 = map(int, recenter_input.split())

            image_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.jpg') or f.endswith('.png')],
                                 key=numerical_sort)

            for img_file in image_files:
                img_path = os.path.join(input_directory, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to load image {img_file}")
                    continue

                transformed_img = recenter_and_rotate(img, x1, y1, x2, y2)
                recentered_img_path = os.path.join(output_directory, img_file)
                cv2.imwrite(recentered_img_path, transformed_img)
                print(f"Recentered and saved: {recentered_img_path}")

        except ValueError:
            print("Invalid input. Please enter four integer values separated by spaces.")
            return

    directory = output_directory

    projected_length = calculate_projected_length(diameter, arc_length_mm)
    overlap_height = (diameter - projected_length) / 2
    num_overlap_imgs = int(overlap_height / projected_length)

    sample_image_path = os.path.join(directory, sorted(os.listdir(directory), key=numerical_sort)[0])
    sample_image = cv2.imread(sample_image_path)
    img_height, img_width = sample_image.shape[:2]

    x1, x2 = 540, 1320
    y1, y2 = int(img_height / 2 - 15), int(img_height / 2 + 15)

    axis_shift_val = 50
    ratio_threshold = 0.55

    shifts, global_avg_shift = process_directory(directory, x1, x2, y1, y2, ratio_threshold,
                                                 axis_shift_left=axis_shift_val, axis_shift_right=axis_shift_val)

    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')],
                         key=numerical_sort)
    num_imgs = len(image_files)

    cv2.namedWindow('Stitched Image')

    cv2.createTrackbar('Shift', 'Stitched Image', int(global_avg_shift * 100), 10000, update_stitching)
    cv2.createTrackbar('Axis Shift Left', 'Stitched Image', axis_shift_val, 100, update_stitching)
    cv2.createTrackbar('Axis Shift Right', 'Stitched Image', axis_shift_val, 100, update_stitching)

    update_stitching(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

```
