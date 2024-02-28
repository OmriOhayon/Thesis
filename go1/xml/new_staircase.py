from PIL import Image
import numpy as np

# Create a 4x4 numpy array for grayscale values
data = np.array([
    [0, 0, 85, 85],
    [0, 0, 85, 85],
    [170, 170, 255, 255],
    [170, 170, 255, 255]
], dtype=np.uint8)

# Create a PIL image from the numpy array
img = Image.fromarray(data, 'L')

# Save the image
img.save('C:\\Users\\Omri\\Desktop\\MSc\Thesis\\Code\\Height_Maps\\stairs_done.png')