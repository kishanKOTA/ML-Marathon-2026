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

colors = ('b','g','r')

# RGB Day
plt.figure()  # CLAUDE FIX: fresh figure so curves don't accumulate across saves
for i,color in enumerate(colors):
    hist_day_rgb = cv2.calcHist([day_img], [i], None, [256], [0,256])
    plt.plot(hist_day_rgb, color=color)  # CLAUDE FIX: plot each channel inside the loop in its own color
plt.title('Day Image Histogram RGB')
plt.savefig(root / "tests/day_hist_rgb.png")  # CLAUDE FIX: path relative to repo root

# HSV Day
hist_day_hsv = cv2.calcHist([day_img_hsv], [1], None, [256], [0,256])
plt.figure()  # CLAUDE FIX
plt.plot(hist_day_hsv, color='b')
plt.title('Day Image Histogram For B Channel')
plt.savefig(root / "tests/day_hist_hsv.png")  # CLAUDE FIX

# RGB Night
plt.figure()  # CLAUDE FIX
for i,color in enumerate(colors):
    hist_night_rgb = cv2.calcHist([night_img], [i], None, [256], [0,256])
    plt.plot(hist_night_rgb, color=color)  # CLAUDE FIX
plt.title('Night Image Histogram RGB')  # CLAUDE FIX: title said V Channel
plt.savefig(root / "tests/night_hist_rgb.png")  # CLAUDE FIX

# HSV Night
hist_night_hsv = cv2.calcHist([night_img_hsv], [1], None, [256], [0,256])
plt.figure()  # CLAUDE FIX
plt.plot(hist_night_hsv, color='b')
plt.title('Night Image Histogram For B Channel')
plt.savefig(root / "tests/night_hist_hsv.png")  # CLAUDE FIX