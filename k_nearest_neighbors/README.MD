## Image segmentation by KNN Algorithm

### Project Report

### Digital Image Processing

# Problem Statement

##### This Project has an analysis of K - Nearest Neighbour Algorithm

##### on MRI scans to segment the tumour.

##### Implementation on: Jupyter lab

```
https://colab.research.google.com/drive/1ZxnCKDQrWmD4Y_fOdRed9wcM7yViVqTi
```
# Program in Brief
```
The Number Program initially asks for the number of clusters to be selected in the K-NN algorithm. Obviously more the clusters more the number of segmented parts in the image.
So the steps Involved in K-NN algorithm:
1) Choosing the number of Clusters
2) Selecting at random K points for centroid, in our case 40 was passed as the number of
neighbors.
3) Assigning each Data point as we say each pixel value closest to the above centroid that
further gives us clusters.
4) Now we compute and place the new centroid for each cluster.
5) On the last step we just do the reassignment of the new nearest centroid and if in any case any new reassignment took place we would reiterate the above process.
```
##### Note: The program saves the newly generated images in the PWD.

# Results
![Results](https://abhinavutkarsh728.s3.amazonaws.com/14%3A38%3A48+C_4.png)
![Results](https://abhinavutkarsh728.s3.amazonaws.com/14%3A38%3A48+C_3.png)
![Results](https://abhinavutkarsh728.s3.amazonaws.com/Unknown.png)
