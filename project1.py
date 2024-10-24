import cv2
import numpy as np
import math

image_path = input()

# Read the input image
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if gray is None:
    print(f"Error: Could not open or find the image at {image_path}")
    exit()

# Apply Gaussian blur with a 9x9 kernel size
gray = cv2.GaussianBlur(gray, (9, 9), 0)

# Binarize the image using Otsu's thresholding
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Close holes with erosion (morphological operation)
kernel = np.ones((9, 9), dtype=np.uint8)  # Use np.uint8 for morphological operations
new_mask = cv2.erode(binary, kernel)

# invert image
inverted_image = cv2.bitwise_not(new_mask)

# Find connected components, print amount (N)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, 4)

# Create a mask to keep only components with radius between 90 and 130
filtered_labels = np.zeros_like(labels, dtype=np.uint8)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    radius = math.sqrt(area / math.pi)
    
    # Filter based on radius
    if 90 <= radius <= 130:
        # Keep this component
        filtered_labels[labels == i] = 255


# Print the number of filtered components (excluding the background)
filtered_num_labels = len(np.unique(filtered_labels)) - 1
print(filtered_num_labels)

def coin_classification(radius):
    if radius > 127:
        return 25
    elif radius > 111:
        return 5
    elif radius > 100:
        return 1
    elif radius > 94:
        return 10
    else:
        return 0

# Calculate the radius assuming the component is circular
for i in range(1, num_labels):
    if(filtered_num_labels == 0):
        break
    x, y, w, h, area = stats[i]
    cx, cy = centroids[i]
    radius = math.sqrt(area / math.pi)
    coin_class = coin_classification(radius)

    #for test cases
    if(coin_class > 0):
        print(f"{cx:.0f} {cy:.0f} {coin_class}")

