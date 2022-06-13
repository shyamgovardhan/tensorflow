#!/usr/bin/env python3

# Shyam Govardhan
# 3 June 2022
# Coursera: Convolutional Neural Networks in Tensorflow
# Week 1: Assignment

import os
import shutil
import random
from shutil import copyfile

root_dir = '/tmp/cats-v-dogs'

if os.path.exists(root_dir):
  shutil.rmtree(root_dir)

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def create_train_val_dirs(root_path):
  training_path = os.path.join(root_path, "training")
  validation_path = os.path.join(root_path, "validation")
  # print("root_path", root_path)
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
      print(f"{fpath} is zero length, so ignoring.")
  # print(copyCount, " files copied to ", destdir)

def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
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

def testTrainValDirs():
  try:
    create_train_val_dirs(root_path=root_dir)
  except FileExistsError:
    print("You should not be seeing this since the upper directory is removed beforehand")

# def testSplitData():
  # split_data("/tmp/PetImages/Cat", "/tmp/cats-v-dogs/training/cats", "/tmp/cats-v-dogs/validation", 0.6)

testTrainValDirs()
# testSplitData()

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