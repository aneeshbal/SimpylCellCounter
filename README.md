# SimpylCellCounter - An Automated Solution for Quantifying Cells in Brain Tissue

SimpylCellCounter (SCC) is a fast, robust and automated method for quantifying cells in brain tissue. SCC is a Python-based algorithm that utilizes the open-source computer vision package OpenCV and a Tensorflow-based convolutional neural network (CNN). SCC achieves high speeds by initially relying mainly on simple computer vision techniques such as binary thresholding and noise filtering. SCC also uses a CNN in order to detect and count overlapping cells, a far more efficient process than traditional watershed methods. 

SCC is also highly-customizable by allowing the user to alter nearly every parameter. These parameters include threshold value, noise filtering levels and the radius of cells to-be-counted. Additionally, the user can custom-train the CNN to best fit their needs. 

Lastly, SCC requires minimal knowledge of Python and can be run in the easy-to-use Google Colab interface. The advantage of using SCC on Colab is that no environments need to be set up, and no packages need to be installed manually.

To learn more about SCC, read the paper here: https://www.nature.com/articles/s41598-020-68138-4


# New Updates!

SimpylCellCounter (SCC) 2.0 is available in "testing" mode. SCC 2.0 introduces a fully deep learning approach to detecting cells, utilizing a U-net/ResNet like neural network architecture trained on ~30 images of fluorescent and brightfield nuclei. At the moment, it is still a work-in-progress, but will be updated frequently. SCC 2.0 was trained using a Keras frontend with a Tensorflow backend... 


# How to Use SCC

### SCC 1.0 Usage
#### SCC provides two options for use: 1) use on your local machine or 2) use on Google Colab (recommended)

If using 1), click [here](examples/1x/local)

If using the recommended method 2), click [here](examples/1x/colab)

### SCC 2.0 Usage

If using SCC 2.0, click [here](examples/2x)


# Contact 
### arguell5@msu.edu and aneesh.s.bal@gmail.com
