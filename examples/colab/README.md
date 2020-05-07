# (1) Create Gmail if you do not have one
#### Ensure a basic familiarity with Google Drive

# (2) Open Files

#### Open `scc_colab` and select the banner at the top of screen `Open in Colab`


# (3) Set Up Colab
#### First begin by selecting `Runtime` then `Change runtime type`, then under `Hardware accelerator` select `GPU`
![](../../icons/Picture1.png)

![](../../icons/Picture2.png)

# (4) Begin Running Code

## (a) Initialize Session
#### This section imports all the relevant packages
#### Note: Tensorflow 1x is used here for accurate inference since the original CNN model was trained on Tf 1.0
![](../../icons/Picture3.png)

## (b) Connect Drive to Colab Session
#### This section makes your google drive accessible to Colab...once this code is run, a link will appear: 
#### - click this link and follow the prompts until you receieve an authorization code. 
#### - paste the authorization code into the Colab notebook prompt and hit `Enter`
