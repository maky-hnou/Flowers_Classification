# Flowers_Classification  
![Python version][python-version]
[![GitHub issues][issues-image]][issues-url]
[![GitHub forks][fork-image]][fork-url]
[![GitHub Stars][stars-image]][stars-url]
[![License][license-image]][license-url]

Flower Species Classifier using TensorFlow.  


## About this repo:  
In this repo, I used TensorFlow to build VGG16 Neural Network and train it from scratch using the *[102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)*, a dataset consisting of 102 flower categories.  


## Content:  

- **categories_names.json:** a json file conaining the flowers/categories names.
- **prepocessing.py:** the code used to preprocess the images.
- **run_training.py:** the code used to launch the training.
- **test.py:** the code used to test the model once it is trained.
- **train.py:** the code used to train the model.
- **utils.py:** a python file containing utils functions.
- **vgg_16.py:** the code used to build the VGG16 model.
- **requirements.txt:** a text file containing the needed packages to run the project.  


## Train and test the model:  

**1. Prepare the environment:**  
*NB: Use python 3+ only.*  
Before anything, please install the requirements by running: `pip3 install -r requirements.txt`.  

**2. Prepare the data:**  
Download the *102 Category Flower Dataset* available via this [link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).  
Extract all the files into a `flower_data/` directory.  
The extracted data into `flower_data/` should be organized as follows:  
`flower_data/` should contain three folders named `train/`, `test/` and `valid/`.  
*Optional:* you can convert all the dataset into npy file by uncommenting [lines 33 and 34](https://github.com/maky-hnou/Flowers_Classification/blob/4a20e5a91cc880e6e573513c829d77d1313f8817/preprocessing.py#L31) of `preprocessing.py`.

**3. Train the VGG16 model:** (*from scratch*)   
To train the model, run `python3 run_training.py`.   
The trained model will be saved to a directory named `model/`.  

**4. Test the model:**  
To test your trained model, run `python3 run_testing.py`. Don't forget to change the image's and the model's paths.

[python-version]:https://img.shields.io/badge/python-3.6+-brightgreen.svg
[issues-image]:https://img.shields.io/github/issues/maky-hnou/Flowers_Classification.svg
[issues-url]:https://github.com/maky-hnou/Flowers_Classification/issues
[fork-image]:https://img.shields.io/github/forks/maky-hnou/Flowers_Classification.svg
[fork-url]:https://github.com/maky-hnou/Flowers_Classification/network/members
[stars-image]:https://img.shields.io/github/stars/maky-hnou/Flowers_Classification.svg
[stars-url]:https://github.com/maky-hnou/Flowers_Classification/stargazers
[license-image]:https://img.shields.io/github/license/maky-hnou/Flowers_Classification.svg
[license-url]:https://github.com/maky-hnou/Flowers_Classification/blob/master/LICENSE
