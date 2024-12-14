import os
import json
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import re

def sanitize_filename(filename):
    # Replace any character that is not alphanumeric, underscore, hyphen, or dot
    return re.sub(r'[^\w\.-]', '_', filename)

# Set the base directory and train/test directories
base_dir = r'D:\dataset\pure_python_genrated_dataset'
output_dir = r'D:\dataset\structhured_dataset'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# Create train and test directories outside the "38_classes" folder
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lists to hold image file paths and labels for the CSV files
train_data = []
test_data = []
class_label_Id_mapping_dict = {}
train_classes = []
test_classes = []

# Counter for unique filenames
unique_counter = 0
class_counter = -1

# Traverse through all subdirectories and files in the base directory
for root, dirs, files in os.walk(base_dir):
    # Filter for image files (e.g., .jpg, .jpeg, .png, .svg)
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.svg'))]
    
    # If there are images in the directory, process them
    if images:
        # Split images into train and test sets
        train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)
        
        # Get the class label from the class directory
        # The class label is the parent directory of the current directory
        class_label = os.path.basename(os.path.dirname(root))
        class_label_sanitized = sanitize_filename(class_label)
        
        # Get the subdirectory name
        sub_dir = os.path.basename(root)
        sub_dir_sanitized = sanitize_filename(sub_dir)
        
        # Process and copy the train images
        for img_name in train_images:
            src_path = os.path.join(root, img_name)
            img_name_sanitized = sanitize_filename(img_name)
            
            if class_label not in train_classes:
                class_counter+=1
                train_classes.append(class_label)
                class_label_Id_mapping_dict[class_counter] = class_label
                dst_dirs = os.path.join(train_dir, f"{class_counter}")
                os.makedirs(dst_dirs, exist_ok=True)

            # Create a unique filename
            unique_filename = f"{train_classes.index(class_label)}/{class_label_sanitized}_{sub_dir_sanitized}_{img_name_sanitized}"

            # Check for filename collision
            dst_path = os.path.join(train_dir, f"{unique_filename}")
            if os.path.exists(dst_path):
                unique_counter += 1
                name, ext = os.path.splitext(unique_filename)
                unique_filename = f"{train_classes.index(class_label)}/{name}_{unique_counter}{ext}"
                dst_path = os.path.join(train_dir, unique_filename)
            
            # Copy the image and record the data
            shutil.move(src_path, dst_path)
            train_data.append({'filename': unique_filename, 'class': class_label, 'classId': train_classes.index(class_label)})
        
        # Process and copy the test images
        for img_name in test_images:
            src_path = os.path.join(root, img_name)
            img_name_sanitized = sanitize_filename(img_name)

            if class_label not in test_classes:
                test_classes.append(class_label)
            
            # Create a unique filename
            unique_filename = f"{class_label_sanitized}_{sub_dir_sanitized}_{img_name_sanitized}"
            
            # Check for filename collision
            dst_path = os.path.join(test_dir, unique_filename)
            if os.path.exists(dst_path):
                unique_counter += 1
                name, ext = os.path.splitext(unique_filename)
                unique_filename = f"{name}_{unique_counter}{ext}"
                dst_path = os.path.join(test_dir, unique_filename)
            
            # Copy the image and record the data
            shutil.move(src_path, dst_path)
            test_data.append({'filename': unique_filename, 'class': class_label, 'classId': test_classes.index(class_label)})

# Create DataFrames for train and test data
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Save the DataFrames to CSV files outside the "38_classes" folder
train_csv_path = os.path.join(output_dir, 'Train.csv')
test_csv_path = os.path.join(output_dir, 'Test.csv')
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print(f"Train and test split completed. CSV files created at {train_csv_path} and {test_csv_path}.")

print(json.dumps(class_label_Id_mapping_dict))