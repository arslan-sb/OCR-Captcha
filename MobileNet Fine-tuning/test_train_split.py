import os
import shutil
import random

# Define paths
data_dir = '/home/arslan/reCaptcha Clusters'
train_dir = '/home/arslan/DIP Projects/MobileNet Fine-tuning/train'
test_dir = '/home/arslan/DIP Projects/MobileNet Fine-tuning/test'

# Ensure the existence of train and test directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

# Set the split ratio
split_ratio = 0.9  # 90% train, 10% test

# Iterate through each class directory in your dataset
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        images = os.listdir(class_dir)
        random.shuffle(images)  # Shuffle images to randomize the split
        num_train = int(len(images) * split_ratio)
        
        # Create directories for the class in train and test sets
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy images to train and test directories based on split ratio
        for i, image in enumerate(images):
            src_path = os.path.join(class_dir, image)
            if i < num_train:
                dst_path = os.path.join(train_class_dir, image)
            else:
                dst_path = os.path.join(test_class_dir, image)
            shutil.copy(src_path, dst_path)

print("Dataset split into train and test sets successfully.")
