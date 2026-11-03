''' 
Purpose: Add day/night and deer visibility tag to images by looking at RGB distributions
and calculating bbox extent (helpful for diagnosis since those two introduce a lot of variability). 

Plan it to use OpenCV + Numpy for manipulation
'''
import cv2
import numpy

from pathlib import Path
from matplotlib import pyplot as plt

# Defines paths that work from wherever the file is called
root = Path(__file__).resolve().parents[1]

test_night_img_path = root / Path("data/raw/images/SSWI000000034464903C.jpg")
test_day_img_path = root / Path("data/raw/images/SSWI000000007580756A.jpg")

night_img = cv2.imread(test_night_img_path, cv2.IMREAD_COLOR)
night_img_hsv = cv2.cvtColor(night_img, cv2.COLOR_BGR2HSV)
night_img_hsv_slice = night_img_hsv[:,:,2]

print("Night image HSV:", night_img_hsv_slice)

day_img = cv2.imread(test_day_img_path, cv2.IMREAD_COLOR)
day_img_hsv = cv2.cvtColor(day_img, cv2.COLOR_BGR2HSV)
day_img_hsv_slice = day_img_hsv[:,:,2]

print("Day image HSV:", day_img_hsv_slice)

hist = cv2.calcHist([day_img_hsv], [2], None, [256], [0,256])

# plot the above histogram
plt.plot(hist, color='b')
plt.title('Image Histogram For V Channel')
plt.savefig("/home/lucasp/projects/research/ML-Marathon-2026/tests/hist.png") 