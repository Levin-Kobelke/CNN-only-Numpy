# CNN-only-Numpy
In this repository I will create and train a basic CNN. But I am only going to use Numpy.
The basic CNN is supposed to be able to solve the Kaggle Dogs vs. Cats challenge sufficently enough as to demonstrate learning.

For personal training and as a comparisson I will build the same network archtecture using the PyTorch libary.
To compare the simple CNN to a more optimal result I fine tuned a pretrained ResNet model by reseting the last fully connected layer and retraining it on the Dogs vs. Cats data. 

The Training and Testing Data is located at "dropbox" link https://www.dropbox.com/sh/ul2m43h9ke4t5vc/AAAp_sEHK9yJp5Wn9noIyLjTa?dl=0
After downloading the folders training_renamed and testing_renamed have to be copied into /numpyNet/data/ where the csv files live.
Optionally, you could also change the "img_dir" argument for the DataLoader to the dir of the testing/training images if you do not want to have the images in the /numpyNet/data/ folder

To train the networks execute the respective three train_* file in the folder numpyNet. 
