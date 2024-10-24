import cv2
import numpy as np
import math

image_path = input()

# Load images
img1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
img2 = cv2.imread('./reference.png', cv2.IMREAD_COLOR)

# Initialize the SIFT detector with a limit on the number of features
sift = cv2.SIFT_create(nfeatures=3000)

# Detect keypoints and compute descriptors
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Reshape descriptors for distance calculation
d1 = descriptors1.reshape((descriptors1.shape[0], 1, descriptors1.shape[1]))
d2 = descriptors2.reshape((1, descriptors2.shape[0], descriptors2.shape[1]))

# Compute Euclidean distances between descriptors
dist = np.sum((d1 - d2) ** 2, axis=2)

# Store matched points
pt1, pt2 = [], []

# Find matches using ratio test and store matched points
for i in range(len(dist)):
    a, b = np.argpartition(dist[i], kth=2)[:2]
    A, B = dist[i, a], dist[i, b]
    if B < A:
        A, B = B, A
        a, b = b, a
    r = A / B
    if r < 0.5:
        x1, y1 = keypoints1[i].pt
        x2, y2 = keypoints2[a].pt
        pt1.append((x1, y1))
        pt2.append((x2, y2))

# Convert points to numpy arrays
pt1 = np.array(pt1)
pt2 = np.array(pt2)

# Parameters for clustering
distance_threshold = 40  # Distance threshold for points to be considered close (in pixels)
min_neighbors = 5  # Minimum number of neighbors within distance_threshold to be part of a dense cluster

# Function to find dense cluster points
def find_dense_cluster(points, distance_threshold, min_neighbors):
    cluster_points = []
    for i, point in enumerate(points):
        distances = np.linalg.norm(points - point, axis=1)
        neighbor_count = np.sum(distances < distance_threshold)
        if neighbor_count >= min_neighbors:
            cluster_points.append(point)
    return np.array(cluster_points)

# Find dense clusters in pt1 and pt2
cluster_points1 = find_dense_cluster(pt1, distance_threshold, min_neighbors)

# Calculate cluster centroid, radius/bounding box height, and angle
centroid1 = 0
radius1 = 0
furthest_left_point = 0

if len(cluster_points1) > 0:
    centroid1 = cluster_points1.mean(axis=0)
    distances1 = np.linalg.norm(cluster_points1 - centroid1, axis=1)
    radius1 = np.mean(distances1)

    # print(f"radius: {int(radius1)}")
    #bounding box height approximation
    if(int(radius1) <= 20):
        height = int(35*radius1)
    elif(int(radius1 <= 40)):
        height = int(25*radius1)
    else:
        height = int(15*radius1)

    # finding the angle
    x_coords = cluster_points1[:, 0]
    y_coords = cluster_points1[:, 1]

    # Compute the coefficients of the line using the least-squares method
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]

    # Determine the start and end points of the line for drawing
    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(m * x_min + b)
    y_max = int(m * x_max + b)

    theta = math.atan(m)  # Angle in radians
    theta_degrees = math.degrees(theta)  # Convert to degrees
    # clockwise rotation
    if(theta_degrees >= -10 or theta_degrees <= 10):
        print(f"{int(centroid1[0])} {int(centroid1[1])} {int(height)} {0}")
    else:
        theta_degrees = 360 - abs(theta_degrees)
        print(f"{int(centroid1[0])} {int(centroid1[1])} {int(height)} {theta_degrees}")

else:
    print("0 0 0 0")



# cx = int(centroid1[0])
# cy = int(centroid1[1])

# cv2.circle(img1, (cx, cy), 10, (255, 255, 0), 10)

# pt1x = cx
# pt1y = cy - int(height/2)
# pt2x = pt1x
# pt2y = pt1y + height
# cv2.line(img1, (pt1x, pt1y),(pt2x, pt2y), (0,255,0), 3)

# img1 = cv2.resize(img1, (int(img1.shape[1] * 0.6), int(img1.shape[0] * 0.6)), interpolation=cv2.INTER_AREA)

# # Display the result until ESC key is pressed
# while cv2.waitKey(0) != 27:
#     cv2.imshow("Matches with Dense Cluster Centroids", img1)
# cv2.destroyAllWindows()

