import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# CONFIGURATION
# --------------------------
IMAGE_PATH = "M162350671LC.tif"   # Input image

# Output file names as requested
OUTPUT_DISPARITY = "disparity_map.png"
OUTPUT_SFS = "sfs_slope_map.png"
OUTPUT_3D_DEM = "final_dem_3d.png"
OUTPUT_2D_SHADED = "final_dem_2d_shaded.png"

# Cropping region (adjust to your needs)
CROP_Y_RANGE = (20000, 21024)
CROP_X_RANGE = (1000, 2024)
STEREO_SHIFT = 10  # Shift for synthetic right image

# --------------------------
# LOAD AND CROP
# --------------------------
def load_and_crop_image(path):
    if not os.path.exists(path):
        print(f"Error: Input file not found at '{path}'")
        sys.exit(1)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read '{path}', check file format.")
        sys.exit(1)

    print(f"Image loaded: {img.shape}, dtype: {img.dtype}")
    cropped = img[CROP_Y_RANGE[0]:CROP_Y_RANGE[1], CROP_X_RANGE[0]:CROP_X_RANGE[1]]
    return cropped

# --------------------------
# DISPARITY MAP
# --------------------------
def generate_disparity_map(left_img, right_img):
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    disparity = stereo.compute(left_img, right_img)
    disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity

# --------------------------
# SFS MAP
# --------------------------
def calculate_sfs_slope(gray_img):
    blurred = gaussian_filter(gray_img, sigma=3)
    gx = sobel(blurred, axis=1)
    gy = sobel(blurred, axis=0)
    slope = np.sqrt(gx ** 2 + gy ** 2)
    slope = np.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
    slope = np.clip(slope, 0, np.max(slope)).astype(np.float32)

    print(f"Slope calculation stats:")
    print(f"  Shape: {slope.shape}, dtype: {slope.dtype}")
    print(f"  Min: {np.min(slope):.3f}, Max: {np.max(slope):.3f}")

    slope_norm = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return slope_norm

# --------------------------
# HYBRID DEM (2D)
# --------------------------
def generate_hybrid_dem(disparity, sfs):
    hybrid = cv2.addWeighted(disparity.astype(np.float32), 0.5,
                             sfs.astype(np.float32), 0.5, 0)
    hybrid = cv2.normalize(hybrid, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return hybrid

# --------------------------
# 3D SURFACE PLOT
# --------------------------
def plot_3d_surface(height_map, save_path):
    X, Y = np.meshgrid(np.arange(height_map.shape[1]), np.arange(height_map.shape[0]))
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, height_map, cmap='terrain', edgecolor='none')
    ax.set_title("Final Hybrid DEM (3D View)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# --------------------------
# 2D SHADED RELIEF
# --------------------------
def plot_2d_shaded_relief(height_map, save_path):
    ls = LightSource(azdeg=315, altdeg=45)
    shaded = ls.shade(height_map.astype(np.float32), cmap=plt.cm.terrain, vert_exag=1.2, blend_mode='soft')
    plt.figure(figsize=(8, 6))
    plt.imshow(shaded)
    plt.axis('off')
    plt.title("Final Hybrid DEM (2D Shaded Relief)")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# --------------------------
# MAIN
# --------------------------
def main():
    left_image = load_and_crop_image(IMAGE_PATH)
    right_image = np.roll(left_image, -STEREO_SHIFT, axis=1)

    # Disparity Map
    disparity_map = generate_disparity_map(left_image, right_image)
    cv2.imwrite(OUTPUT_DISPARITY, disparity_map)
    print(f"Disparity map saved as {OUTPUT_DISPARITY}")

    # SFS Map
    sfs_map = calculate_sfs_slope(left_image)
    cv2.imwrite(OUTPUT_SFS, sfs_map)
    print(f"SFS slope map saved as {OUTPUT_SFS}")

    # Hybrid DEM
    hybrid_dem = generate_hybrid_dem(disparity_map, sfs_map)

    # 3D Plot
    plot_3d_surface(hybrid_dem, OUTPUT_3D_DEM)
    print(f"3D DEM saved as {OUTPUT_3D_DEM}")

    # 2D Shaded Relief
    plot_2d_shaded_relief(hybrid_dem, OUTPUT_2D_SHADED)
    print(f"2D shaded relief saved as {OUTPUT_2D_SHADED}")

if __name__ == "__main__":
    main()
