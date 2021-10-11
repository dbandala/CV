# Clase-Estereoscopica

## Poincloud Processing

This file contains the program that allows a multi-view reconstruction.

The program works in the following way:

1. Searches a specified folder for all the .PLY files and imports them into the workspace.
2. Orderds all the .PLY files with a given camera ID sequence.
3. Global registration using the extrinsic parameters, within the camera transformation matrix.<br>
    
    The Camera Transformation Matrix: The transformation that places the camera in the correct position and orientation in world space (this is the transformation that you would apply to a 3D model of the camera if you wanted to represent it in the scene).
* The View Matrix: This matrix will transform vertices from world-space to view-space.  This matrix is the inverse of the camera’s transformation matrix.
* The view matrix is the inverse of the camera’s transformation matrix in world-space. The rotation must also be inverted.

4. ICP registration takes place.
5. Poisson surface reconstruction takes place.
6. Outputs a mesh of the target in a .PLY file