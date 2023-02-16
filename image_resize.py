from PIL import Image
import os

# Set the directory containing the images
directory = "/path/to/folder"

# Set the desired width and height
width = 800
height = 600

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg"):
        # Open the image file
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path)

        # Resize the image
        resized_image = image.resize((width, height))

        # Save the resized image with the same filename
        resized_image.save(image_path)

        print(f"{filename} resized to {width}x{height}")
