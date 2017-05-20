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
    * Keras Functional API
    * Using pretrained models
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



### In memory: loading  & resizing the data

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
Now the data is pretty much ready to be fed to a Convolutional Neural Network!

### Out-of-memory: loading, resizing and saving the data

Sometimes we cannot work with the data in-memory because of various reasons - we are unable to load the full data into memory, we prefer to load it chunk-by-chunk, so most of our RAM is free, enabling us to perform different operations while training a model.

Here we must approach the situation in a bit different way - we will load each image, resize it and then save into a destination folder, so it can then be easily fed to a neural net without further manipulation done by hand.

```
import os
import time
import glob
import cv2
import pandas as pd
import numpy as np
from scipy import misc
from PIL import Image



def make_dirs(src, dst):
    labels = os.listdir(src)
    os.chdir(dst)
    for i in labels:
        if i not in os.listdir(dst):
            os.mkdir(i)
    return

def load_resize_save_train(src, dst):
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
                final = Image.fromarray((img).astype(np.uint8))
                final.save(dst + flbase)
            except Exception as e:
                print(e, 'Failed to process:', fl)
    print('Processed train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return



# Now we specify source and destination folder. Then we create directiories based on directory names in original folder and we process the data.

src = 'data/training_data/'
dst = 'data/processed_training_data/'
make_dirs(src, dst)
load_resize_save_train(src, dst)
```

So what does this function do? It works in a way very similar to the one used for loading the data into memory but this time we load, resize and then save the image into our destination folder. And yes, we use PIL here, because when using OpenCV's function to save the image loaded in RGB, we'd have colors mixed up in saved images. 

* src <- stands for source folder
* dst <- stands for destination folder
* flbase <- in flbase we've got class and image name contained
* dst + flbase <- dst and flbase concatenated enables us to save an image to a proper class folder. name of the image is not changed

then we'll have our image saved into a folder signifying a proper class (this is important when we'll be loading the data later - most of data loaders takes folders as basis for class distinguishment.


## Real-time data augmentation with Keras
------

Working on images, we cannot perform feature engineering as-is, so we have to direct our gaze onto different methods. Here comes the data augmentation, which, as stated in [Deep Learning Book 12.2](http://www.deeplearningbook.org/contents/applications.html) is a method for data preprocessing, where we apply random transformations to the data, creating slightly modified copies of images. This should help the model achieve better generalization, as it becomes more invariant to small changes of the input.
Fortunately for us all, users of Keras, there are functions enabling you to perform real-time images augmentation with a few (quite) easy steps.
The biggest change is the fact that when using those augmentation methods, we use generators to feed the data to the network.

```
import keras

keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

This shows how many various augmentations methods we can apply to our data. To better understand what each of them does, head to [Keras Image Processing documentation](https://keras.io/preprocessing/image/).


### ImageDataGenerator in-memory (.flow)

```
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 111)

train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.3,
                rotation_range=180,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
train_datagen.fit(X, augment = True)
valid_datagen = ImageDataGenerator(rescale=1./255,)

model.fit_generator(train_datagen.flow(X_tr, y_tr, batch_size = batch_size), 
                        steps_per_epoch = X_tr.shape[0]/batch_size,
                        validation_data = valid_datagen.flow(X_val, y_val, 
                        batch_size = batch_size, shuffle = False),
                        validation_steps = X_val.shape[0]/batch_size, epochs = 1)
```

Here we have an example of using augmentation based on data loaded into memory. First, we create subsets for training and validation, then we apply random augmentations to training data, while keeping validation data not augmented (we could also do that, but in general, if we are not going to create augmented test set, unaugmented validation data will provide us with a more reliable indication of model performance).

We need to use fit_generator to feed the data into model, where we specify each of the parameters according to the [fit_generator documentation](https://keras.io/models/model/).


### ImageDataGenerator from disk (.flow_from_directory)

```
train_src = 'data/train_split/'
valid_src = 'data/valid_split/'

batch_size = 32
img_size = (299, 299)
train_samples_number = len(os.listdir(train_src))
validation_samples_number = len(os.listdir(valid_src))
classes = ['Type_1', 'Type_2', 'Type_3']

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255,)

train_generator = train_datagen.flow_from_directory(
        train_src,
        target_size=img_size,
        batch_size=batch_size,
        shuffle = True,
        classes=classes,
        class_mode='categorical')
validation_generator = valid_datagen.flow_from_directory(
        valid_src,
        target_size=img_size,
        batch_size = batch_size,
        shuffle = False,
        classes=classes,
        class_mode='categorical')

model.fit_generator(train_generator,steps_per_epoch = X_tr.shape[0]/batch_size,
                        validation_data = validation_generator, 
                        validation_steps = X_val.shape[0]/batch_size, epochs = 1)
```

Approach for using ImageGenerator with __.flow_from_directory__ is very similar, except for the fact that here we have to specify folder paths containing training and validation splits.
In order to split, you can randomly choose a subset of data and move chosen files into validation folder (based on [sudo's NCFM script](https://github.com/dcrush/Nature-Conservancy-Kaggle/blob/master/1_vgg16_augmented.ipynb))

```
import os
import time
import glob

data_src = 'data/'

print('Splitting dataset.')
t = time.time()
os.chdir(data_src)
os.makedirs('train_split')
os.makedirs('valid_split')
copytree('full_data', 'train_split')
os.chdir('{}/train_split'.format(data_src))
for cls in glob.glob('*'): os.mkdir('../valid_split/' + cls)
random_subset = np.random.permutation(glob.glob('*/*.jpg'))
for i in range(200): os.rename(random_subset[i], '../valid_split/' + random_subset[i])
print('Split done, it took: ', time.time() - t)
```

Folders 'train_split' and 'valid_split' will be created, then from 'full_data' folder all files will be copied to 'train_split'. Afterwards we create folders in 'valid_split' corresponding to folder names in training folder, take a random subset (200 images) of data and move them using 'os.rename' from 'train_split' folder to 'valid_split' folder. 


## Convolutional Neural Networks in Keras
------

### Creating models from scratch

When creating models from scratch, there are two ways to do this - using Sequential API and Functional API.
The second one is much more powerful and interesting, I'll thus dedicate a bit more of space to it.

### Sequential API
[Sequential API](https://keras.io/getting-started/sequential-model-guide/) enables you to stack layers on top of one another, thus creating a model consisting of a straight sequence of layers. It is good for basic usage but especially for CNN's I'd recommend learning Functional API.


Here's an example of a simple CNN model created with Sequential API:

```
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation

def sequential_cnn(num_classes, img_size):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=img_size))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
    
img_size = (299, 299, 3)
num_classes = 3

model = sequential_cnn(num_classes, img_size)
model.fit(X_tr, y_tr, validation_data = (X_val, y_val), batch_size = 8, epochs = 10)
```


### Functional API

The other way is to use [Functional API](https://keras.io/getting-started/functional-api-guide/) where we create models defining their 'branches' and then appending each new layer to a chosen branch. It's much more powerful way to create models, enabling us to merge input models in many different ways (for example feeding one Convolutional branch with image input and a second MLP one with categorical/numerical features). 

This can be useful not only for CNN models but also for RNN's, where we will often not only have raw text features but also can derive characteristics of a text and feed those into MLP (bonus example will be provided ;).

Almost all of current state-of-the-art CNN models are branched in some way, take a look at [ResNet- Models/Visualization](https://github.com/KaimingHe/deep-residual-networks) or [Inception (here Inception-v4 & Inception-ResNet)](https://arxiv.org/pdf/1602.07261.pdf) architectures. 

Keras Functional API enables us to create complicated models easily and, what's also very important, easily modify pretrained models, which are often the best choice if we'd like to achieve maximum performance in a competition.


And the same model as above, created using Functional API:

```
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Activation

def functional_cnn(num_classes, img_size):
    
    input_layer = Input(img_size)
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes)(x)
    output_layer = Activation('softmax')(x)
    
    model = Model(input = input_layer, output = output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
    
img_size = (299, 299, 3)
num_classes = 3

model = sequential_cnn(num_classes, img_size)
model.fit(X_tr, y_tr, validation_data = (X_val, y_val), batch_size = 8, epochs = 10)
```

First, we define 'input_layer', where shape of provided images is specified. Then, a branch 'x' is created, to which all further layers are added. When model architecture is stated, in 'Model' we define the input layer and output layer. 

I recommend taking a look at [Keras applications on github](https://github.com/fchollet/keras/tree/master/keras/applications) where Inception v3 and ResNet50 are defined. 
Now we can smoothly proceed to working and manipulation pretrained Keras models such as Inception and ResNet mentioned above.



### Keras Pretrained Models

We can train a model from scratch every time we encounter a new problem, sure. In most cases this is computationally expensive, requires a lot of time and quite a big dataset to achieve good results.
Fortunately, usually best or at least very good results can be achieved with simply fine-tuning already trained model.

I won't delve into details, when should this be done as [Andrej Karpathy provides a very useful guideline for using pretrained models](http://cs231n.github.io/transfer-learning/). __It's worth reading!__ 
On the other hand, I would always recommend at least giving a few pretrained models a try because even with significantly different data you may still be able to achieve at least a bit better result than with a model trained from scratch.

Functional API comes in very handy when tuning pretrained models.

Here are a few steps how to make them work:
1. Pick a pretrained models, in my case - ResNet50:

    * Do you want to include top dense layers? If so - *include_top = True*, if not: *include_top = False*.
    This is important if you'd like to change size of input images, when top is included, you can only feed the model with size corresponding to original training size, here: (224, 224).
    * If you'd like to feed the net with images of different size, then don't include the top layers and specify your shape in *input_shape*. I'd like to feed mine with 3-channel RGB images with image format corresponding to Tensorflow's, so I'll use *input_shape = (299, 299, 3)*, for Theano format it should be *input_shape = (3, 299, 299)*.
      
2. Get specified layer of the model:

    * *get_layer(index = n).output*. Where *n* indicates which layer index from the last you will pick. 
    * You can also pick layers by name with *get_layer(name = layer_name).output*. Best to take a look at: [get_layer documentation](https://keras.io/models/model/).
    
    To be sure which exactly you'll be picking, take a look at [Keras's ResNet architecture](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py).
    
3. Add as many layers as you like and create as many branches as you'd like.

4. Compile the model.


Let's take a look at examples which will be described below:

```
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adadelta, Adam, SGD, Nadam, RMSprop
from keras.layers import merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Model

def resnet_simple():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                            input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output

    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(num_classes, activation='softmax', name = 'predictions')(output)

    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model


def resnet_2branches():

    learning_rate = 0.0001
    optimizer = SGD(lr=learning_rate, momentum = 0.9, decay = 1e-3, nesterov = True)
    
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                        input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -2).output
    
    glob_pool = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
    max_pool = MaxPooling2D((7, 7), strides=(7, 7), name='max_pool')(output)
    concat_pool = merge([glob_pool, max_pool], mode='concat', concat_axis=-1)
    output = Flatten(name='flatten')(concat_pool)
    output = Dropout(0.2)(output)
    output = Dense(1024)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.2)(output)
    output = Dense(1024)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(num_classes, activation='softmax', name = 'predictions')(output)
    
    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model

size = (299, 299, 3)
num_classes = 3
```



In *resnet_simple* we simply add a few layers to the model. Because we get last layer with *.get_layer*, the one that will be last in our new model is:
```
x = AveragePooling2D((7, 7), name='avg_pool')(x)
```
based on model architecture.


In *renset_2branches* we would like to add two different pooling layers, so that each can learn a bit different features:
```
glob_pool = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(output)
max_pool = MaxPooling2D((7, 7), strides=(7, 7), name='max_pool')(output)
```
Afterwards we concatenate their outputs on *concat_axis = -1* because that's the axis for RGB channels in TF format.
```
concat_pool = merge([glob_pool, max_pool], mode='concat', concat_axis=-1)
```
and feed them to a dense layer:
```
output = Flatten(name='flatten')(concat_pool)
```

which we assume will be able to distinguish between classes based on their different features, each learned from a different Pooling layer.
Now we need to get penultimate layer from the ResNet50 architecture, which according to [Keras's ResNet architecture](https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py) is 
```
x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
```
because otherwise we would pool from a AveragePooling layer and that's not something we want to do.




### Extracting Features from CNN's

Sometimes we would like to extract features from a Convolutional Neural Network and feed them to a different classifier (like our beloved XGBoost) hoping it may learn to classify better. Features can also be extracted to what the network learns. Here I would also recommend great Andrej Karpathy's CS231n notes - [Visualizing what ConvNets learn](http://cs231n.github.io/understanding-cnn/).

[Keras Documentation](https://keras.io/applications/) specifies that we can use *get_layer('block4_pool').output* to get a proper layer for feature extraction:

#### Extract features from an arbitrary intermediate layer with VGG19
```
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

This is a method that works perfectly on a model defined in keras.applications, where you have named layers.
When you want to extract features from a layer based on it's layer_index, then provide an argument specifying that you extract based on *index* not *name*. That is method 1.
Second method is to use *layers[layer_index]* and provide a corresponding *layer_index*.

* __Method 1:__
```
model = Model(input = orig_model.input, output = orig_model.get_layer(index = n).output)
```


* __Method 2:__
```
model = Model(input = orig_model.input, output = orig_model.layers[layer_index].output)
```


__It is much easier to visualize that on an example,__ so let's give *resnet_simple()* another mention here.
You will usually want to extract features for further classification, when you have an already trained model, so let's begin with loading it's weights, then creating a feature extractor and feeding it with data to extract features.

Model is for visualization purpose only:

```
def resnet_simple():
    resnet_notop = ResNet50(include_top=False, weights='imagenet',
                            input_tensor=None, input_shape = size)
    output = resnet_notop.get_layer(index = -1).output

    output = Flatten(name='flatten')(output)
    output = Dropout(0.2)(output)
    output = Dense(512)(output)
    output = PReLU()(output)
    output = BatchNormalization()(output)
    output = Dropout(0.3)(output)
    output = Dense(num_classes, activation='softmax', name = 'predictions')(output)

    Resnet_model = Model(resnet_notop.input, output)
    Resnet_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return Resnet_model
```


```
from keras.models import load_model, Model

n = -5
orig_model = load_model(checks_src + 'resnet_simple_weights.h5')
feature_extractor = Model(input = orig_model.input, output = orig_model.get_layer(index = n).output) # Method 1
feature_extractor = Model(input = orig_model.input, output = orig_model.layers[n].output) # Method 2
```

We want to extract features from 
```
output = PReLU()(output)
```

layer and it's 5th from the bottom of the model. Here are indexes corresponding to layers visualized:

```
output = Flatten(name='flatten')(output)
output = Dropout(0.2)(output)
output = Dense(512)(output) # -5 index
output = PReLU()(output) # -4 index
output = BatchNormalization()(output) # -3 index
output = Dropout(0.3)(output) # -2 index
output = Dense(num_classes, activation='softmax', name = 'predictions')(output) # -1 index
```

When a model is built, now we predict with it data we want to feed the classifier with using:
```
train_features = feature_extractor.predict(X_train, batch_size = 16)
valid_features = feature_extractor.predict(X_valid, batch_size = 16)
```

because our *Dense()* layer is 512-neurons wide, our *train_features* will be a Numpy array of size __MxN__, where __M__ is the number of training examples and __N__  size of the layer output, so here __N = 512__.


Here comes the final step - feeding the features into XGB classifier:

```
import xgboost as xgb
from sklearn.metrics import log_loss

params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.03,
        'objective': 'reg:linear',
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': num_class,
        'max_depth': 12,
        'min_child_weight': 100,
        'booster': 'gbtree',
        
        }

num_class = 3

d_train = xgb.DMatrix(train_features, label=y_tr)
d_valid = xgb.DMatrix(valid_features, label=y_val)
watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params, d_train, 100000, watchlist, early_stopping_rounds = 50,
               verbose_eval = 50)
preds_val = clf.predict(xgb.DMatrix(xgb_val), ntree_limit=clf.best_ntree_limit)
print('Logloss:', log_loss(y_val, preds_val))
```



#### So the full code for feature extraction will look like this:


```
import xgboost as xgb
from keras.models import load_model, Model
from sklearn.metrics import log_loss

n = -5

orig_model = load_model(checks_src + 'resnet_simple_weights.h5')
feature_extractor = Model(input = orig_model.input, output = orig_model.get_layer(index = n).output) 
train_features = feature_extractor.predict(X_train, batch_size = 16)
valid_features = feature_extractor.predict(X_valid, batch_size = 16)


params = {
        'seed': 0,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.03,
        'objective': 'reg:linear',
        'eval_metric': 'mlogloss',
        'objective': 'multi:softprob',
        'num_class': num_class,
        'max_depth': 12,
        'min_child_weight': 100,
        'booster': 'gbtree',
        
        }

num_class = 3

d_train = xgb.DMatrix(train_features, label=y_tr)
d_valid = xgb.DMatrix(valid_features, label=y_val)
watchlist = [(d_train, 'train'), (d_valid, 'eval')]

clf = xgb.train(params, d_train, 100000, watchlist, early_stopping_rounds = 50,
               verbose_eval = 50)
preds_val = clf.predict(xgb.DMatrix(xgb_val), ntree_limit=clf.best_ntree_limit)
print('Logloss:', log_loss(y_val, preds_val))
```

__Voila!__ Now we can see if performance of our new classifier trained on CNN features is better than that of an CNN itself.
