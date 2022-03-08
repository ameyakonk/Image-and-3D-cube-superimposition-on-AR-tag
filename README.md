# Image-and-3D-cube-superimposition-on-AR-tag

## Overview
The repository contains the following contents.

1. Fast fourier transform to filter an image
2. AR tag detection and id identification using Homography matrix.
3. Image superimposition on the AR tag.
4. 3D cube superimposition on the AR tag. 

## Personnel
### Ameya Konkar 

UID:118191058

Master's Student at University of Maryland, College Park

## Dependencies 

Install the following dependencies:

1.  Install Anaconda Navigator to create multiple environments for future
    projects. Anaconda navigator also gives you access to Spyder and Jupyter
    Notebook, very useful tools for python coding. For windows, refer here. For
    Linux, refer [here](https://docs.anaconda.com/anaconda/install/linux/).
2.  Install required packages onto your virtual environment. Replace “myenv”
    with your environment name. Enter the following commands in your
    terminal window. Press ‘y’ when prompted. (Step h just launches
    the spyder application). Remember to always work in your virtual
    environment to properly run your codes .
    a. conda create -n myenv python=3.7
    b. conda activate myenv
    c. conda install -c conda-forge opencv=4.1.0
    d. conda install -c anaconda numpy
    e. conda install -c conda-forge matplotlib
    f. conda install -c conda-forge imutils
 
### Building the Program and Tests

```
sudo apt-get install git
git clone --recursive https://github.com/ameyakonk/Image-and-3D-cube-superimposition-on-AR-tag.git
cd <path to repository>
conda activate <name of env>
```

Q1 a. To Run the code:
```
cd src/
chmod +x FFT.py
python FFT.py 
```

Q1 b. To Run the code:  
```
cd src/
chmod +x AR_detection.py
python AR_detection.py

```   
Q2 a. To Run the code:
```
cd src/
chmod +x ImageSuperImpose.py
python ImageSuperImpose.py 
```
Q2 b. To Run the code:
```
cd src/
chmod +x Cube_code_detection.py
python Cube_code_detection.py 
```

