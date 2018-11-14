# using anaconda environment pymc_py36
from __future__ import print_function, division
#from builtins import range, input
import argparse
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
#import matplotlib.pyplot as plt
from IPython.display import display
import h5py
from PIL import Image

from glob import glob
parser = argparse.ArgumentParser(description='keras bloodcell example')

parser.add_argument('-i', '--train', help = "The location of the train datasets", required = True)
parser.add_argument('--val', required=True,
                        help='location of the validate')

parser.add_argument('--epoch', required=True,
                        help='how many epoch , default to 10')

parser.add_argument('--output', required=True,
                        help='output path uploads model weights')
parser.add_argument('--load_model', required=True,
                        help='load previously trained model weights and continue trainig? True/False')


args = parser.parse_args()


# re-size all the images to this
IMAGE_SIZE = [197, 197] # feel free to change depending on dataset

# training config:
epochs = int(args.epoch)
batch_size = 32

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = args.train
valid_path = args.val


# useful for getting number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# useful for getting number of classes
folders = glob(train_path + '/*')


# look at an image uncomment
#plt.imshow(image.load_img(np.random.choice(image_files)))
#plt.show()


# add preprocessing layer to the front of restnet50
res = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in res.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(res.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)



# create a model object
model = Model(inputs=res.input, outputs=prediction)
from keras.models import model_from_yaml
from keras.models import model_from_json
# serialize model to JSON
import os
# view the structure of the model
#model.summary()
if args.load_model:
    # load json and create model

    json_file = open(args.output+'best_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(args.output+"best_model.h5")
    print("Loaded model from disk")

    
model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)



# create an instance of ImageDataGenerator
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)


# test generator to see how it works and some other useful things

# get label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)
labels = [None] * len(test_gen.class_indices)
for k, v in test_gen.class_indices.items():
  labels[v] = k

# should be a strangely colored image (due to VGG weights being BGR)
"""
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  #plt.show()
  break
"""

# create generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


# fit the model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)



def get_confusion_matrix(data_path, N):
  # we need to see the data in the same order
  # for both predictions and targets
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


#cm = get_confusion_matrix(train_path, len(image_files))
#print(cm)
#valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
#print(valid_cm)

from keras.models import model_from_yaml
from keras.models import model_from_json

# serialize model to JSON
import os
model_json = model.to_json()
#### check os.expand
with open(args.output+ "model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(args.output+"model.h5")
print("Saved model to disk")

#print(os.path.expandvars('$AZ_BATCHAI_JOB_MOUNT_ROOT/afs/'+"model.h5"))
# later...
 
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")

