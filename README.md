# Image classifier project
# Data Scientist Nanodegree

• ```Challenge```: Udacity Data Scientist Nanodegree project for deep learning module titled as 'Image Classifier with Deep Learning' attempts to train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice we had to train this classifier, then export it for use in our application. We had used a dataset (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

• ```Solution```: Used torchvision to load the data. The dataset is split into three parts, training, validation, and testing. For the training, applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. Also need to load in a mapping from category label to category name. Wrote inference for classification after training and testing the model. Then processed a PIL image for use in a PyTorch model. 

• ```Result```: Using the following software and Python libraries: Torch, PIL, Matplotlib.pyplot, Numpy, Seaborn, Torchvision. Thus, achieved an accuracy of 80% on test dataset as a result of above approaches. Performed a sanity check since it's always good to check that there aren't obvious bugs despite achieving a good test accuracy. Plotted the probabilities for the top 5 classes as a bar graph, along with the input image.

### Software and Libraries
This project uses the following software and Python libraries: <br>
NumPy, pandas, Sklearn / scikit-learn, Matplotlib (for data visualization), Seaborn (for data visualization)


### Code File
Othis project is done using both spyder and jupyter notebook imageclassifierproject.py

Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu
