import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, sobel
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm

# --- Configuration ---
# All main settings are in this section for easy modification.
IMAGE_PATH = 'C:/Desktop/Images/M162350671LC_pyr.tif'
CROP_Y_RANGE = (20000, 21024)
CROP_X_RANGE = (1000, 2024)
STEREO_SHIFT = 10  # Horizontal shift in pixels to simulate a stereo image.
STEREO_NUM_DISPARITIES = 64
STEREO_BLOCK_SIZE = 15
SFS_BLUR_SIGMA = 3
FUSION_WEIGHT_STEREO = 0.6 # Weight for the stereo component in the fusion.
FUSION_WEIGHT_SFS = 0.4 # Weight for the SFS component in the fusion.
FINAL_SMOOTH_SIGMA = 1.5

def load_and_crop_image(path, y_range, x_range):
    """Loads an image and crops a specific region."""
    if not os.path.exists(path):
        print(f"Error: File not found at '{path}'")
        return None
    
    # Load the image in grayscale.
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Failed to load the image.")
        return None

    print(f"Image loaded successfully. Original shape: {img.shape}")
    
    # Crop a region from the image.
    cropped_img = img[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    return cropped_img

def calculate_disparity(left_img, right_img):
    """Calculates the disparity map from two stereo images."""
    stereo = cv2.StereoBM_create(numDisparities=STEREO_NUM_DISPARITIES, blockSize=STEREO_BLOCK_SIZE)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    
    # Normalize the disparity map to 0-255 range for visualization.
    disparity_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print("Disparity map generated.")
    cv2.imwrite("disparity_map.png", disparity_visual)
    return disparity_visual

def calculate_sfs_slope(image):
    """Estimates the slope from image brightness gradients (Shape-from-Shading)."""
    # Calculate gradients using a Sobel filter.
    blurred = gaussian_filter(image, sigma=SFS_BLUR_SIGMA)
    dx = sobel(blurred, axis=1)  # Change in the x-direction
    dy = sobel(blurred, axis=0)  # Change in the y-direction
    
    # Calculate the total slope magnitude using the hypotenuse.
    slope = np.hypot(dx, dy)
    
    # Normalize the slope map for visualization.
    slope_norm = cv2.normalize(slope, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    print("SFS slope map generated.")
    cv2.imwrite("sfs_slope_map.png", slope_norm)
    return slope_norm

def plot_3d_dem(dem_data, title="3D Digital Elevation Model"):
    """Creates a 3D surface plot of the DEM data."""
    X, Y = np.meshgrid(range(dem_data.shape[1]), range(dem_data.shape[0]))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot.
    ax.plot_surface(X, Y, dem_data, cmap='terrain', rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_zlabel("Elevation (normalized)")
    
    plt.savefig("final_dem_3d.png", dpi=300)
    print("3D plot saved as 'final_dem_3d.png'")
    plt.show()

def plot_2d_shaded_relief(dem_data, title="Shaded Relief Map"):
    """Creates a 2D shaded relief view of the DEM with a lighting effect."""
    # Set up the light source.
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Apply shading to the DEM.
    shaded_map = ls.shade(dem_data, cmap=cm.terrain, vert_exag=1.2, blend_mode='soft')

    plt.figure(figsize=(10, 8))
    plt.imshow(shaded_map)
    plt.title(title)
    plt.axis('off')
    
    plt.savefig("final_dem_2d_shaded.png", dpi=300)
    print("2D shaded relief map saved as 'final_dem_2d_shaded.png'")
    plt.show()

def main():
    """The main function that orchestrates the entire process."""
    
    # Step 1: Load and crop the source image.
    left_image = load_and_crop_image(IMAGE_PATH, CROP_Y_RANGE, CROP_X_RANGE)
    if left_image is None:
        return # Exit if the image fails to load.

    # Step 2: Simulate a stereo pair from the single cropped image.
    # The original crop will serve as the 'left' image.
    # Shift the 'left' image horizontally to create the 'right' image.
    right_image = np.roll(left_image, -STEREO_SHIFT, axis=1)
    
    # Step 3: Generate the disparity map using Stereo Vision.
    disparity_map = calculate_disparity(left_image, right_image)
    
    # Step 4: Generate the slope map using Shape-from-Shading.
    sfs_map = calculate_sfs_slope(left_image)
    
    # Step 5: Fuse the two maps to create a hybrid DEM.
    # First, normalize both maps to a 0-1 range.
    disp_norm = disparity_map.astype(np.float32) / 255.0
    sfs_norm = sfs_map.astype(np.float32) / 255.0
    
    # Fuse them using a weighted average.
    fused_dem = (FUSION_WEIGHT_STEREO * disp_norm) + (FUSION_WEIGHT_SFS * sfs_norm)
    print("DEMs fused successfully.")
    
    # Step 6: Apply a Gaussian filter to smooth the final DEM and reduce noise.
    fused_smooth_dem = gaussian_filter(fused_dem, sigma=FINAL_SMOOTH_SIGMA)
    
    # Step 7: Plot the final Digital Elevation Model.
    plot_3d_dem(fused_smooth_dem, title="Final Hybrid DEM (3D View)")
    plot_2d_shaded_relief(fused_smooth_dem, title="Final Hybrid DEM (2D Shaded Relief)")

# Standard Python entry point to run the script.
if __name__ == "__main__":
    main()
