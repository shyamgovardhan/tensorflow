#!/usr/bin/env python
# coding: utf-8

# # Week 1: Using CNN's with the Cats vs Dogs Dataset
# 
# Welcome to the 1st assignment of the course! This week, you will be using the famous `Cats vs Dogs` dataset to train a model that can classify images of dogs from images of cats. For this, you will create your own Convolutional Neural Network in Tensorflow and leverage Keras' image preprocessing utilities.
# 
# You will also create some helper functions to move the images around the filesystem so if you are not familiar with the `os` module be sure to take a look a the [docs](https://docs.python.org/3/library/os.html).
# 
# Let's get started!

# In[1]:


import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


# Download the dataset from its original source by running the cell below. 
# 
# Note that the `zip` file that contains the images is unzipped under the `/tmp` directory.

# In[2]:


# If the URL doesn't work, visit https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765
# And right click on the 'Download Manually' link to get a new URL to the dataset

# Note: This is a very large dataset and will take some time to download

get_ipython().system('wget --no-check-certificate      "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"      -O "/tmp/cats-and-dogs.zip"')

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()


# Now the images are stored within the `/tmp/PetImages` directory. There is a subdirectory for each class, so one for dogs and one for cats.

# In[3]:


source_path = '/tmp/PetImages'

source_path_dogs = os.path.join(source_path, 'Dog')
source_path_cats = os.path.join(source_path, 'Cat')


# os.listdir returns a list containing all files under the given path
print(f"There are {len(os.listdir(source_path_dogs))} images of dogs.")
print(f"There are {len(os.listdir(source_path_cats))} images of cats.")


# **Expected Output:**
# 
# ```
# There are 12501 images of dogs.
# There are 12501 images of cats.
# ```

# You will need a directory for cats-v-dogs, and subdirectories for training
# and validation. These in turn will need subdirectories for 'cats' and 'dogs'. To accomplish this, complete the `create_train_val_dirs` below:

# In[4]:


import os

# Define root directory
root_dir = '/tmp/cats-v-dogs'

# Empty directory to prevent FileExistsError is the function is run several times
if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

# GRADED FUNCTION: create_train_val_dirs
def create_train_val_dirs(root_path):
  """
  Creates directories for the train and test sets
  
  Args:
    root_path (string) - the base directory path to create subdirectories from
  
  Returns:
    None
  """  

  ### START CODE HERE

  # HINT:
  # Use os.makedirs to create your directories with intermediate subdirectories
  # Don't hardcode the paths. Use os.path.join to append the new directories to the root_path parameter

  # os.makedirs(root_dir)
  training_path = os.path.join(root_path, "training")
  validation_path = os.path.join(root_path, "validation")
  print("root_path", root_path)
  for subdir in ("cats", "dogs"):    
    os.makedirs(os.path.join(training_path, subdir))
    os.makedirs(os.path.join(validation_path, subdir))

  ### END CODE HERE

  
try:
  create_train_val_dirs(root_path=root_dir)
except FileExistsError:
  print("You should not be seeing this since the upper directory is removed beforehand")


# In[5]:


# Test your create_train_val_dirs function

for rootdir, dirs, files in os.walk(root_dir):
    for subdir in dirs:
        print(os.path.join(rootdir, subdir))


# **Expected Output (directory order might vary):**
# 
# ``` txt
# /tmp/cats-v-dogs/training
# /tmp/cats-v-dogs/validation
# /tmp/cats-v-dogs/training/cats
# /tmp/cats-v-dogs/training/dogs
# /tmp/cats-v-dogs/validation/cats
# /tmp/cats-v-dogs/validation/dogs
# 
# ```

# Code the `split_data` function which takes in the following arguments:
# - SOURCE_DIR: directory containing the files
# 
# - TRAINING_DIR: directory that a portion of the files will be copied to (will be used for training)
# - VALIDATION_DIR: directory that a portion of the files will be copied to (will be used for validation)
# - SPLIT_SIZE: determines the portion of images used for training.
# 
# The files should be randomized, so that the training set is a random sample of the files, and the validation set is made up of the remaining files.
# 
# For example, if `SOURCE_DIR` is `PetImages/Cat`, and `SPLIT_SIZE` is .9 then 90% of the images in `PetImages/Cat` will be copied to the `TRAINING_DIR` directory
# and 10% of the images will be copied to the `VALIDATION_DIR` directory.
# 
# All images should be checked before the copy, so if they have a zero file length, they will be omitted from the copying process. If this is the case then your function should print out a message such as `"filename is zero length, so ignoring."`. **You should perform this check before the split so that only non-zero images are considered when doing the actual split.**
# 
# 
# Hints:
# 
# - `os.listdir(DIRECTORY)` returns a list with the contents of that directory.
# 
# - `os.path.getsize(PATH)` returns the size of the file
# 
# - `copyfile(source, destination)` copies a file from source to destination
# 
# - `random.sample(list, len(list))` shuffles a list

# In[6]:


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
  """
  Splits the data into train and test sets
  
  Args:
    SOURCE_DIR (string): directory path containing the images
    TRAINING_DIR (string): directory path to be used for training
    VALIDATION_DIR (string): directory path to be used for validation
    SPLIT_SIZE (float): proportion of the dataset to be used for training
    
  Returns:
    None
  """

  ### START CODE HERE  
  def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

  def create_train_val_dirs(root_path):
    training_path = os.path.join(root_path, "training")
    validation_path = os.path.join(root_path, "validation")
    print("root_path", root_path)
    for subdir in ("cats", "dogs"):    
      os.makedirs(os.path.join(training_path, subdir))
      os.makedirs(os.path.join(validation_path, subdir))

  def validateInput(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    if not os.path.exists(SOURCE_DIR):
      print(SOURCE_DIR, ": directory does not exist")
      return False
    if not os.path.exists(TRAINING_DIR):
      print(TRAINING_DIR, ": directory does not exist")
      return False
    if not os.path.exists(VALIDATION_DIR):
      print(VALIDATION_DIR, ": directory does not exist")
      return False
    if not (isfloat(SPLIT_SIZE) and SPLIT_SIZE > 0 and SPLIT_SIZE < 1):
      print(SPLIT_SIZE, ": 0 > SPLIT_SIZE > 1")
      return False
    return True

  def copySampleFiles(fileset, srcdir, destdir):
    copyCount = 0
    for fname in fileset:
      fpath = os.path.join(srcdir, fname)
      destinationPath = os.path.join(destdir, fname)
      if not os.path.getsize(fpath) == 0:
        # print("Copying ", fpath, " to ", destinationPath)
        copyfile(fpath, destinationPath)
        copyCount += 1
      else:
        print(f"{fname} is zero length, so ignoring.")
    # print(copyCount, " files copied to ", destdir)
  
  if not validateInput(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    return

  srcFiles = os.listdir(SOURCE_DIR)
  srcFileCount = len(srcFiles)
  random.sample(srcFiles, srcFileCount)
  sampleSize = round(srcFileCount * SPLIT_SIZE)
  trainingFiles = slice(0, sampleSize)
  validationFiles = slice(sampleSize, srcFileCount)
  trainingFileCount = len(srcFiles[trainingFiles])
  validationFileCount = len(srcFiles[validationFiles])

  # print(SOURCE_DIR, ": file count: ", srcFileCount)
  # print("sampleSize: ", sampleSize)
  # print("trainingFileCount: ", trainingFileCount)
  # print("validationFileCount: ", validationFileCount)
  copySampleFiles(srcFiles[trainingFiles], SOURCE_DIR, TRAINING_DIR)
  copySampleFiles(srcFiles[validationFiles], SOURCE_DIR, VALIDATION_DIR)
  ### END CODE HERE


# In[7]:


# Test your split_data function

# Define paths
CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"

TRAINING_DIR = "/tmp/cats-v-dogs/training/"
VALIDATION_DIR = "/tmp/cats-v-dogs/validation/"

TRAINING_CATS_DIR = os.path.join(TRAINING_DIR, "cats/")
VALIDATION_CATS_DIR = os.path.join(VALIDATION_DIR, "cats/")

TRAINING_DOGS_DIR = os.path.join(TRAINING_DIR, "dogs/")
VALIDATION_DOGS_DIR = os.path.join(VALIDATION_DIR, "dogs/")

# Empty directories in case you run this cell multiple times
if len(os.listdir(TRAINING_CATS_DIR)) > 0:
  for file in os.scandir(TRAINING_CATS_DIR):
    os.remove(file.path)
if len(os.listdir(TRAINING_DOGS_DIR)) > 0:
  for file in os.scandir(TRAINING_DOGS_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_CATS_DIR)) > 0:
  for file in os.scandir(VALIDATION_CATS_DIR):
    os.remove(file.path)
if len(os.listdir(VALIDATION_DOGS_DIR)) > 0:
  for file in os.scandir(VALIDATION_DOGS_DIR):
    os.remove(file.path)

# Define proportion of images used for training
split_size = .9

# Run the function
# NOTE: Messages about zero length images should be printed out
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, VALIDATION_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, VALIDATION_DOGS_DIR, split_size)

# Check that the number of images matches the expected output
print(f"\n\nThere are {len(os.listdir(TRAINING_CATS_DIR))} images of cats for training")
print(f"There are {len(os.listdir(TRAINING_DOGS_DIR))} images of dogs for training")
print(f"There are {len(os.listdir(VALIDATION_CATS_DIR))} images of cats for validation")
print(f"There are {len(os.listdir(VALIDATION_DOGS_DIR))} images of dogs for validation")


# **Expected Output:**
# 
# ```
# 666.jpg is zero length, so ignoring.
# 11702.jpg is zero length, so ignoring.
# ```
# 
# ```
# There are 11250 images of cats for training
# There are 11250 images of dogs for training
# There are 1250 images of cats for validation
# There are 1250 images of dogs for validation
# ```

# Now that you have successfully organized the data in a way that can be easily fed to Keras' `ImageDataGenerator`, it is time for you to code the generators that will yield batches of images, both for training and validation. For this, complete the `train_val_generators` function below.
# 
# Something important to note is that the images in this dataset come in a variety of resolutions. Luckily, the `flow_from_directory` method allows you to standarize this by defining a tuple called `target_size` that will be used to convert each image to this target resolution. **For this exercise, use a `target_size` of (150, 150)**.
# 
# **Hint:** 
# 
# Don't use data augmentation by setting extra parameters when you instantiate the `ImageDataGenerator` class. This will make the training of your model to take longer to reach the necessary accuracy threshold to pass this assignment and this topic will be covered in the next week.

# In[8]:


# GRADED FUNCTION: train_val_generators
def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
  """
  Creates the training and validation data generators
  
  Args:
    TRAINING_DIR (string): directory path containing the training images
    VALIDATION_DIR (string): directory path containing the testing/validation images
    
  Returns:
    train_generator, validation_generator - tuple containing the generators
  """
  ### START CODE HERE

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  train_datagen = ImageDataGenerator( rescale = 1.0/255. )

  # Pass in the appropiate arguments to the flow_from_directory method
  train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                      batch_size=20,
                                                      class_mode='binary',
                                                      target_size=(150, 150))

  # Instantiate the ImageDataGenerator class (don't forget to set the rescale argument)
  validation_datagen = ImageDataGenerator( rescale = 1.0/255. )

  # Pass in the appropiate arguments to the flow_from_directory method
  validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                batch_size=20,
                                                                class_mode='binary',
                                                                target_size=(150, 150))
  ### END CODE HERE
  return train_generator, validation_generator


# In[9]:


# Test your generators
train_generator, validation_generator = train_val_generators(TRAINING_DIR, VALIDATION_DIR)


# **Expected Output:**
# 
# ```
# Found 22498 images belonging to 2 classes.
# Found 2500 images belonging to 2 classes.
# ```
# 

# One last step before training is to define the architecture of the model that will be trained.
# 
# Complete the `create_model` function below which should return a Keras' `Sequential` model.
# 
# Aside from defining the architecture of the model, you should also compile it so make sure to use a `loss` function that is compatible with the `class_mode` you defined in the previous exercise, which should also be compatible with the output of your network. You can tell if they aren't compatible if you get an error during training.
# 
# **Note that you should use at least 3 convolution layers to achieve the desired performance.**

# In[10]:


from tensorflow.keras.optimizers import RMSprop # accuracy 95% not reached
from tensorflow.keras.optimizers import Adam    # Accuracy 95% reached

# GRADED FUNCTION: create_model
def create_model():
  # DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
  # USE AT LEAST 3 CONVOLUTION LAYERS

  ### START CODE HERE

  model = tf.keras.models.Sequential([ 
    tf.keras.layers.Conv2D(128, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
  ])

  
  model.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']) 
    
  ### END CODE HERE

  return model


# Now it is time to train your model!
# 
# **Note:** You can ignore the `UserWarning: Possibly corrupt EXIF data.` warnings.

# In[11]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.95 and logs.get('val_accuracy') >= 0.80):
      print("Training Accuracy >= 95% and Validation Accuracy >= 80% , so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()  

# Get the untrained model
model = create_model()

# Train the model
# Note that this may take some time.
history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks=[callbacks])


# Once training has finished, you can run the following cell to check the training and validation accuracy achieved at the end of each epoch.
# 
# **To pass this assignment, your model should achieve a training accuracy of at least 95% and a validation accuracy of at least 80%**. If your model didn't achieve these thresholds, try training again with a different model architecture and remember to use at least 3 convolutional layers.

# In[12]:


#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()


# You will probably encounter that the model is overfitting, which means that it is doing a great job at classifying the images in the training set but struggles with new data. This is perfectly fine and you will learn how to mitigate this issue in the upcoming week.
# 
# Before downloading this notebook and closing the assignment, be sure to also download the `history.pkl` file which contains the information of the training history of your model. You can download this file by running the cell below:

# In[13]:


def download_history():
  import pickle
  from google.colab import files

  with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

  files.download('history.pkl')

download_history()


# You will also need to submit this notebook for grading. To download it, click on the `File` tab in the upper left corner of the screen then click on `Download` -> `Download .ipynb`. You can name it anything you want as long as it is a valid `.ipynb` (jupyter notebook) file.

# **Congratulations on finishing this week's assignment!**
# 
# You have successfully implemented a convolutional neural network that classifies images of cats and dogs, along with the helper functions needed to pre-process the images!
# 
# **Keep it up!**
