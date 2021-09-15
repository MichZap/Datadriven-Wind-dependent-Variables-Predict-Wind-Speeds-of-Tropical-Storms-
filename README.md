# Datadriven-Wind-dependent-Variables-Predict-Wind-Speeds-of-Tropical-Storms-
10th place solution to drivendata.org competition https://www.drivendata.org/competitions/72/predict-wind-speeds/

Problem description(drivendata):

In this challenge, you will estimate the wind_speed of a storm in knots at a given point in time using satellite imagery. The training data consist of single-band satellite images from 494 different storms in the Atlantic and East Pacific Oceans, along with their corresponding wind speeds. These images are captured at various times throughout the life cycle of each storm. Your goal is to build a model that outputs the wind speed associated with each image in the test set.

For each storm in the training and test sets, you are given a time-series of images with their associated relative time since the beginning of the storm. Your models may take advantage of the temporal data provided for each storm up to the point of prediction. Keep in mind that the goal of this competition is to produce an operational model that uses recent images to estimate future wind speeds.


Approach:

Ensemble of best performing individual models which use both temporal and spacial data inputs.

Score Root-mean-square Error : 6.8585

1. Model:
Torchvisions 18 layer Resnet3D using the last 24 satellite images and information on ocean location and time.
In addition for the final prediction an ensemble of the 5 best checkpoints in combination of 6 folds of the best checkpoint with randomly rotated satellite imagery was used. 

Score Root-mean-square Error : 7.1589

2. Model:
EfficientNetB0 where the RGB channels were used for the last three caputred images, afterwards 9 hour running average was applied to smooth over the predicted wind speeds.
Again a 3 fold rotation ensemble was used to generate the final predictions

Score Root-mean-square Error : 7.5947

Further Comments:

If the .ipynb files do not render you might use https://nbviewer.jupyter.org/ in order to inspect them.

Model 1 was trained as a juypter Notebook on Google Colab and Gradient Paperspace. 
The orignial ipython files aren't accessible anymore, however under HU_ResNet 3D_18.py  you can find the initial code.

Also by mistake there was a version of model 1 trained using different input sizes this is included in Create Submission-r3d_18-v2.ipynb as well for training this model only the respective train and validation transforms need to by adjusted from 112x112 to 114x114


