from PIL import Image

def get_image_dimensions(image_path):
    """
    Get the dimensions of the image.
    
    Parameters:
    image_path (str): The path to the image file.

    Returns:
    tuple: A tuple containing the width and height of the image.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def resize_image(image_path, output_path, new_width, new_height):
    """
    Resize the image to new dimensions.
    
    Parameters:
    image_path (str): The path to the image file.
    output_path (str): The path to save the resized image.
    new_width (int): The new width for the resized image.
    new_height (int): The new height for the resized image.

    Returns:
    None
    """
    with Image.open(image_path) as img:
        resized_img = img.resize((new_width, new_height))
        resized_img.save(output_path)

if __name__ == "__main__":
    # Example usage:
    image_path = "/home/arslan/reCaptcha Clusters/2/1aa9c2.jpg"
    output_path = "path/to/save/resized/image.jpg"
    
    # Get image dimensions
    width, height = get_image_dimensions(image_path)
    print(f"Original dimensions: {width}x{height}")
    usr=input("RESIZE?: ")
    if usr=="y":
        # Resize image if needed
        new_width = int(input("width: "))
        new_height = int(input("Height: "))
        resize_image(image_path, output_path, new_width, new_height)
        print(f"Resized image saved to {output_path}")
