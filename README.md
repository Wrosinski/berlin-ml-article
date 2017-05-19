# Why CNN's (and images in general) don't bite - a quick guide to image processing competitions in Python
-------


Image data is a type of unstructured data, which requires a bit different approach to, let's say, a dataset consisting of numerical and categorical features. Although there are many tutorials explaining different parts of a pipeline to work with such data, it is often harder to find one, in which a whole process will be explained.

When you start working with image datasets, at first it is easy to get lost in all the details:
* Which library for image processing should I pick?
* How should I properly load the data?
* How should I process the data?
* What kind of model should I build? Should I train it from scratch or use a pretrained one?
* How can I perform 'feature engineering' on image data?

and many more.

Fortunately for us all, who have some experience with this kind of data and even more fortunately for those, who wish to begin their adventure with images - there are many tools **significantly** facilitating not only processing but also modeling of such data.
On the other hand, the field of Computer Vision and the part of Deep Learning field connected with Convolutional Neural Networks are both vast and I will not be able to cover many of interesting topics which still apply to working with images.
But let's not waste more time for an introduction!


## Overview:
------

* Image processing tools
    * OpenCV-Python 
    * SciPy
    * Pillow
* Methods for loading and processing data
    * Working with datasets in-memory and out-of-memory
    * Real-time data augmentation with Keras
* Convolutional Neural Networks in Keras:
    * Creating and training from scratch
    * Using pretrained models
    * Keras Functional API
    * Extracting features from models and feeding them into a different model 

    
## Image Processing Tools
-----

### OpenCV - Python

First comes the OpenCV library, which is a library dating back to 1999 and being expanded since then. Currently it consists of many algorithms useful for Computer Vision and Machine Learning. It's [Python Documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/) is worth taking a look at, even just to convince yourself, of how many different things it is capable.
Key features are:
* OpenCV is built on C/C++ and a wrapper for Python is created, therefore enabling the algorithms to retain their original C/C++ speed.
* Images loaded and processed with OpenCV are always in Numpy array format, so it makes it very easy to create a pipeline of operations not only consisting of OpenCV algorithms but also various array manipulations

### SciPy

[Scipy](https://docs.scipy.org/doc/) is a stack of modules for mathematical, scientific and engineering operations, according to their website. And we've got a [scipy.ndimage](https://docs.scipy.org/doc/scipy/reference/ndimage.html) module for multi-dimensional image processing. Maybe there aren't as many possible operations to be executed on images with this library, but for most cases it will certainly be enough.
What's more important, scipy also **loads and processes the arrays in a Numpy format**, thus making it easy to work with.

### Pillow

The last library I'd like to mention is [PIL - Python Imaging Library](https://pillow.readthedocs.io/en/4.1.x/) which idea is more similar to OpenCV's, as it is a library made strictly for image processing methods. For some operations it's functions is more easily usable (like cropping) but unfortunately it defines it's own Image class, so you are forced to convert between this and Numpy arrays whenever you'd like to perform a Numpy operation.

#### Summary

Because of Pillow's Image class, for most usage scenarios connected with loading or processing data in simple ways, I would recommend SciPy or OpenCV. And if you'd like to have nice colors, when displaying images in your notebooks, then I would go for Scipy (I will explain why exactly a bit later).

## Loading & Processing Image Data
-----

### Installing the libraries:

#### OpenCV

```
pip install opencv-python
```

#### Scipy

```
pip install scipy
```

#### Pillow

```
pip install Pillow
```



### Loading  & resizing the data

Usually the most comfortable way to work with image data is to simply load it into memory, performing operations on it either during loading (image by image) or after the whole data is loaded.

How to approach loading the data? Oftentimes we won't work on full-size images. Why? Because of memory constraints, the fact that bigger image usually doesn't provide a performance boost for CNN and lowers batch size (this frequently causes longer training time) and also the fact that when making use of pretrained models, we often have to resize the images to size, which the images had during first training.

#### OpenCV

```
import cv2

path = 'images/img01.jpg'
size = (299, 299)

img = cv2.imread(path) # cv2 uses BGR as default color ordering
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # if we would like to have nicely-looking images when displayed
img = cv2.resize(img, size)
```

#### Scipy

```
from scipy import ndimage
from scipy import misc # misc has functions for loading and resizing images

path = 'images/img01.jpg'
size = (299, 299)

img = scipy.misc.imread(path, mode = 'RGB') # in scipy we can specify color ordering during loading. nice!
img = scipy.misc.imresize(img, size)
```

#### Pillow

```
from Pillow import Image

path = 'images/img01.jpg'
size = (299, 299)

img = Image.open(path) # returns Image class instance
img = img.resize(size)
```

Now we have a resized image. That's nice, certainly, but as we see, we have to specify a path to load it. So what should we do if we have a few folders, and there are many images in every of those?
There are many ways to do that. I'll propose one I've been using which has been inspired by [ZFTurbo's kernel](https://www.kaggle.com/zfturbo/fishy-keras-lb-1-25267).

```
import os
import time
import glob
import cv2
import pandas as pd
import numpy as np
from scipy import misc

def load_train(src):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()
    print('Read train images')
    folders = ['Type_1', 'Type_2', 'Type_3']
    for i, fld in enumerate(folders):
        print('Load folder {}'.format(fld))
        path = os.path.join(src, fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = fld + '/' + os.path.basename(fl)
            try:
                img = misc.imread(fl, mode = 'RGB')
                img = misc.imresize(img, (299, 299))
            except Exception as e:
                print(e, 'Failed to load:', fl)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(i)
    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    y_train = np.array(y_train)
    return X_train, y_train, X_train_id

src = 'data/training_data/'
X, y, train_ids = load_train(src)
```

This way we have data loaded, with classes being read from folders (according to an assumption that images belonging to different classes are in different folders) and filenames for all images contained in train_ids.
