# Smile-Detector
Using deep learning to detect smiles from a live video feed.
## Descreption
This project was made for detecting smile in a live video using the webcam or a pre-recorded video.<br>
## Preview

## Requirements
- numpy
- matplotlib
- openCV
- sklearn
- imutils
- TensorFlow/Theano
- Keras
- h5py

You can install all the required libraries by running the following command 
##### pip install requirements.txt
## Functionalities
1. Filter to detect face. 
2. CNN for traning the model(Using LeNet architecture).

## Procedure
1. Gather a dataset of people smiling and not smiling which is tighly cropped around the face.
2. Train the Convolutional Neural Network using `train_model.py` and your custom data set(the network is designed to handle class imbalance).
3. Run the `detect_smile.py` to start a live video using your webcam, or you can also supply an optional argument to enter the path to a video. <br>
## Credits
Adrian Rosebrock creator of [PyimageSearch](https://www.pyimagesearch.com) 
