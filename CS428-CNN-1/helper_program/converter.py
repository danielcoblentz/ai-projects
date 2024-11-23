from PIL import Image
import os

def convert_to_black_and_white(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            # Open an image file
            with Image.open(img_path) as img:
                # Convert the image to grayscale
                bw = img.convert('L')
                # Create output path
                output_path = os.path.join(output_folder, filename)
                # Save the new image
                bw.save(output_path)

# Example usage
input_folder = '/Users/dancoblentz/Desktop/CS428-CNN-1/datasets/no_filter_hand_gesture_dataset/nine'
output_folder = '/Users/dancoblentz/Desktop/CS428-CNN-1/datasets/filtered_hand_gesture_dataset/nine'
convert_to_black_and_white(input_folder, output_folder)
