import cv2
import math
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("../data/_aJOs5B9T-Q/00146.png")

# Define the keypoints as a list of tuples
keypoints = [(100, 100), (200, 100), (150, 200), (100, 300), (200, 300)]

# Define the angles between keypoints
angles = [30, 60, 45, 90, 120]

# Draw the keypoints and angles on the image using OpenCV
for i, point in enumerate(keypoints):
    # Draw the keypoint
    cv2.circle(image, point, 5, (0, 0, 255), -1)

    # Draw the angle text
    angle_text = str(angles[i]) + "°"
    cv2.putText(image, angle_text, (point[0] + 10, point[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw the angle line
    angle_rad = math.radians(angles[i])
    line_end = (int(point[0] + 50 * math.cos(angle_rad)), int(point[1] + 50 * math.sin(angle_rad)))
    cv2.line(image, point, line_end, (0, 255, 0), 2)

# Display the image using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()