# Lunar Digital Elevation Model (DEM) Generation

This project was developed for the Bhartiya Antariksh Hackathon 2025. It demonstrates a method to generate a high-resolution Digital Elevation Model (DEM) of the lunar surface from a single high-resolution satellite image.

The core of this project is a **hybrid model** that combines two computer vision techniques—**Stereo Vision** and **Shape-from-Shading (SFS)**—to create a more robust and detailed terrain map than either method could produce alone.

## Methodology

The process works as follows:
1.  **Simulated Stereo Pair**: A single grayscale lunar image is loaded, and a stereo pair (left and right images) is simulated by creating a slightly shifted version of the original image.
2.  **Stereo Vision**: OpenCV's `StereoBM` algorithm is used on the simulated pair to calculate a disparity map, which provides a baseline estimation of depth.
3.  **Shape-from-Shading (SFS)**: The brightness gradients of the original image are analyzed using a Sobel filter to estimate the surface slope, providing fine textural details.
4.  **Hybrid Fusion**: The depth map from Stereo Vision and the slope map from SFS are combined using a weighted average to create a single, fused DEM.
5.  **Visualization**: The final DEM is visualized as both a 3D terrain model and a 2D shaded relief map using Matplotlib.

## Technologies Used
- **Python**
- **OpenCV** for image processing and stereo vision algorithms.
- **NumPy** for numerical operations and array manipulation.
- **SciPy** for filtering and image processing functions.
- **Matplotlib** for creating the final 2D and 3D visualizations.

## How to Use

1.  **Get a Lunar Image**: Download a high-resolution grayscale lunar image (e.g., a `.tif` or `.png` file). This project was tested with TMC (Terrain Mapping Camera) data.
2.  **Update the Script**: Open the Python script and update the `IMAGE_PATH` variable to point to the location of your downloaded image file.
    ```python
    # Update this line with the path to your image
    IMAGE_PATH = 'path/to/your/lunar_image.tif'
    ```
3.  **Run the Script**: Execute the Python script. The program will process the image and generate several output files in the same directory:
    - `disparity_map.png`: The depth map from the stereo algorithm.
    - `sfs_slope_map.png`: The slope map from the SFS algorithm.
    - `final_dem_3d.png`: The final 3D visualization of the terrain.
    - `final_dem_2d_shaded.png`: The final 2D shaded relief map.

---
*This project is a submission for the Bhartiya Antariksh Hackathon 2025 and is intended for educational and research purposes.*
