﻿# Custom Multiclass Classification from Crawling Images and Training Custom Model to Predicting Classification 

## Overview
This project is divided into 2 parts.
I. Using Selenium to crawl hundreds of google images of desired keyword 
II. Use those images to train your multiclass classification model. 



### I. Crawling
Upon running, enter the desired keyword to be searched on google images.
Run this concurrently or multiple times to also crawl different keyword images (for multiclass).



### II. Custom Multiclass Classification Model
Run preprocess_data.py to preprocess the images you have crawled from 'I. Crawling'.
This will organize the images into train, val, test directories.



## Image Crawling

```
python crawling.py
```

## Train Custom Image Classification 

```
cd multiclass
python preprocess_data.py
python classifciation.py
```

## Predict Unseen Image's Classification 

```
cd multiclass
python predict.py
```


# Experiment and Result

## Crawling
Ran crawling.py 5 times using the following keywords (for 200 images per keyword)

"camel, duck, horse, penguins, puffins"

## Multiclass Classification
  <img src="/result.png" width="500" title="Confusion Matrix">


## Prediction
Choose an image of your own to test your classification on the model you have built. It will predict which class it belongs to.



#### Envrionment / Used Framework and Libraries
conda env

- tensorflow 2.4
- h5py 2.1 
- selenium 
- urllib
- matplotlib
- numpy
- pandas
- seaborn
