# SimpylCellCounter - An Automated Solution for Quantifying Cells in Brain Tissue

SimpylCellCounter (SCC) is a fast, robust and automated method for quantifying cells in brain tissue. SCC is a purely Python-based algorithm that utilizes the open-source computer vision package OpenCV and a Tensorflow-based convolutional neural network (CNN). SCC achieves high speeds by initially relying mainly on simple computer vision techniques such as binary thresholding and noise filtering. SCC also uses a CNN in order to detect and count overlapping cells, a far more efficient process than traditional watershed methods. 

SCC is also highly-customizable by allowing the user to alter nearly every parameter. These parameters include threshold value, noise filtering levels and the radius of cells to-be-counted. Additionally, the user can custom-train the CNN to best fit their needs. 

Lastly, SCC requires minimal knowledge of Python and can be run in the easy-to-use Google Colab interface. The advantage of using SCC on Colab is that no environments need to be set up, and no packages need to be installed manually.


# How to Use SCC

### For the easiest way to use SCC, click [here](examples)


# Contact 
### arguell5@msu.edu and aneesh.s.bal@gmail.com
