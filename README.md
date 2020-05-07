# SimpylCellCounter
SimpylCellCounter - An Automated Solution for Quantifying Cells in Brain Tissue
SimpylCellCounter (SCC) is a fast, robust and automated method for quantifying cells in brain tissue. SCC is a purely Python-based algorithm that utilizes the open-source computer vision package OpenCV and a Tensorflow-based convolutional neural network (CNN). SCC achieves high speeds by initially relying mainly on simple computer vision techniques such as binary thresholding and noise filtering. SCC also uses a CNN in order to detect and count overlapping cells, a far more efficient process than traditional watershed methods. 

SCC is also highly-customizable by allowing the user to alter nearly every parameter. These parameters include threshold value, noise filtering levels and the radius of cells to-be-counted. Additionally, the user can custom-train the CNN to best fit their needs. 

Lastly, SCC requires minimal knowledge of Python and can be run in the easy-to-use Google Colab interface. The advantage of using SCC on Colab is that no environments need to be set up, and no packages need to be installed manually.

#### First, you will need to install Python and utilize Jupyter Notebook to use SimpylCellCounter (SCC)...For steps on how to do this, open the PowerPoint "SCC.pptx"

#### After installation is complete, download the folder above "source". Do NOT rearrange any of the files in this folder as they are all necessary to ensure proper processing!

#### For specific instructions on how to navigate the code for SCC, open the PowerPoint "SCC.pptx"

If you are interested in recreating aspects of SCC, go to the folder "recreationFunctions" to find out more.



##### Contact: arguell5@msu.edu and aneesh.s.bal@gmail.com
