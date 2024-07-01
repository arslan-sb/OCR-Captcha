from PIL import Image
import os

def resize_images(input_dir, output_dir, size):
    """
    Resize all images in the input directory and save them to the output directory.

    :param input_dir: Directory containing the images to be resized.
    :param output_dir: Directory to save the resized images.
    :param size: Tuple indicating the size to resize to, e.g., (width, height).
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Construct the full file path
        file_path = os.path.join(input_dir, filename)

        # Open an image file
        with Image.open(file_path) as img:
            # Resize image
            resized_img = img.resize(size)

            # Save resized image to the output directory
            output_path = os.path.join(output_dir, filename)
            resized_img.save(output_path)

# Example usage:
input_directory = '/home/arslan/reCaptcha Clusters/2'
output_directory = '/home/arslan/reCaptcha Clusters/2r'
resize_to = (200, 50)  # Example size

resize_images(input_directory, output_directory, resize_to)
