import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm

# === STEP 1: Load your TMC image ===
image_path = 'C:/Desktop/Images/M162350671LC_pyr.tif'

if not os.path.exists(image_path):
    print("❌ File not found.")
    exit()

img = cv2.imread(image_path, 0)
if img is None:
    print("❌ Failed to load image.")
    exit()

print("✅ Image loaded:", img.shape)

# === STEP 2: Crop a meaningful region ===
# Crop a 1024 x 1024 region from center area
crop = img[20000:21024, 1000:2024]  # (y range, x range)
cv2.imwrite("left.png", crop)

# === STEP 3: Simulate Stereo by shifting horizontally ===
right = crop[:, 10:]
right = cv2.copyMakeBorder(right, 0, 0, 10, 0, cv2.BORDER_REPLICATE)
cv2.imwrite("right.png", right)

# === STEP 4: Disparity Map using StereoBM ===
left_img = cv2.imread("left.png", 0)
right_img = cv2.imread("right.png", 0)

stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_vis = np.uint8(disp_vis)
cv2.imwrite("disparity_map.png", disp_vis)

# === STEP 5: Simulated SFS using brightness gradients ===
blurred = gaussian_filter(left_img, sigma=3)
dx = sobel(blurred, axis=1)
dy = sobel(blurred, axis=0)
slope = np.hypot(dx, dy)

if np.isnan(slope).any() or np.isinf(slope).any():
    print("❌ Slope contains invalid values.")
    exit()

slope_min, slope_max = np.min(slope), np.max(slope)
if slope_max - slope_min == 0:
    print("❌ Invalid slope range.")
    exit()

slope_norm = ((slope - slope_min) / (slope_max - slope_min)) * 255
sfs_vis = slope_norm.astype(np.uint8)
cv2.imwrite("sfs_slope_map.png", sfs_vis)

# === STEP 6: Hybrid Fusion (Stereo + SFS) ===
disp_norm = (disp_vis.astype(np.float32) - disp_vis.min()) / (disp_vis.max() - disp_vis.min())
sfs_norm = (sfs_vis.astype(np.float32) - sfs_vis.min()) / (sfs_vis.max() - sfs_vis.min())

# Weighted fusion
fused = 0.6 * disp_norm + 0.4 * sfs_norm

# === STEP 7: Smooth the fused DEM ===
fused_smooth = gaussian_filter(fused, sigma=1.5)

# === STEP 8: 3D Plot of Terrain ===
X, Y = np.meshgrid(range(fused_smooth.shape[1]), range(fused_smooth.shape[0]))
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, fused_smooth, cmap='terrain', linewidth=0, antialiased=True)
plt.title("Final Output: Smoothed Simulated Hybrid DEM")
plt.tight_layout()
plt.savefig("fused_dem_smooth.png", dpi=300)
plt.show()

# === Optional: Lighting-based 2D terrain view ===
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(fused_smooth, cmap=cm.terrain, vert_exag=1, blend_mode='soft')

plt.figure(figsize=(10, 6))
plt.imshow(rgb)
plt.title("Hybrid DEM (2D View with Lighting)")
plt.axis('off')
plt.tight_layout()
plt.savefig("fused_lit_dem.png", dpi=300)
plt.show()
