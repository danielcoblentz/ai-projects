from PIL import Image
import os

def convert_to_black_and_white(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            with Image.open(img_path) as img:
                # single-channel grayscale, this builds the filtered dataset that --filter 2 in train_model.py trains on
                bw = img.convert('L')
                output_path = os.path.join(output_folder, filename)
                bw.save(output_path)

# example usage, run once per gesture class folder
input_folder = '/Users/dancoblentz/Desktop/CS428-CNN-1/datasets/no_filter_hand_gesture_dataset/nine'
output_folder = '/Users/dancoblentz/Desktop/CS428-CNN-1/datasets/filtered_hand_gesture_dataset/nine'
convert_to_black_and_white(input_folder, output_folder)
