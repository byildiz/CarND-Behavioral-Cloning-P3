import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.callbacks import ModelCheckpoint
#from keras.utils import plot_model

def read_logs(data_paths):
  '''
  Reads all log files line by line
  '''
  lines = []
  for data_path in data_paths:
    with open(data_path + 'driving_log.csv', 'r') as file:
      reader = csv.reader(file)
      for line in reader:
        if line[0] == 'center':
          # print headers for debugging
          print(line)
          # don't add headers
          continue
        line = [data_path + 'IMG/' + col.strip().split('/')[-1] if i < 3 else col.strip() for i, col in enumerate(line)]
        lines.append(line)
  return lines

def read_image(path):
  '''
  Reads the image in RGB
  '''
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) 

def read_data(lines, angle_correction):
  '''
  Reads data and returns corresponding X and y
  '''
  images = []
  steerings = []
  for line in lines:
    # load center image in RGB
    img = read_image(line[0])
    steering = float(line[3])

    # add center image
    images.append(img)
    steerings.append(steering)

    # add flipped of center image
    images.append(np.fliplr(img))
    steerings.append(-steering)

    # load left image
    left_img = read_image(line[1])
    left_steering = steering + angle_correction

    # add left image
    images.append(left_img)
    steerings.append(left_steering)

    # add flipped of left image
    #images.append(np.fliplr(left_img))
    #steerings.append(-left_steering)

    # load right image
    right_img = read_image(line[2])
    right_steering = steering - angle_correction

    # add right image
    images.append(right_img)
    steerings.append(right_steering)

    # add flipped of right image
    #images.append(np.fliplr(right_img))
    #steerings.append(-right_steering)

  # convert them to numpy arrays and return them
  return np.array(images), np.array(steerings)

"""## Data Generators"""

def generate_samples(lines):
  '''
  This function generates some possible variations of moments for using in batch_generator
  '''
  samples = []
  for line in lines:
    samples.append({'line': line, 'image': 'center', 'flipped': False})
    samples.append({'line': line, 'image': 'center', 'flipped': True})
    samples.append({'line': line, 'image': 'left', 'flipped': False})
    samples.append({'line': line, 'image': 'left', 'flipped': True})
    samples.append({'line': line, 'image': 'right', 'flipped': False})
    samples.append({'line': line, 'image': 'right', 'flipped': True})
  return samples

def batch_generator(samples, angle_correction, batch_size=32):
  '''
  Reads batch sized data each time it is called
  '''
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      steerings = []
      for sample in batch_samples:
        path_index = 0
        if sample['image'] == 'left':
          path_index = 1
        elif sample['image'] == 'right':
          path_index = 2
        
        # load image in RGB
        img = read_image(sample['line'][path_index])
        steering = float(sample['line'][3])
        
        # correct steering angle if frame does not come from center camera
        if path_index != 0:
          if path_index == 1:
            steering += angle_correction
          else:
            steering -= angle_correction
        
        if sample['flipped']:
          img = np.fliplr(img)
          steering = -steering
        
        images.append(img)
        steerings.append(steering)
      yield np.array(images), np.array(steerings)

# Models
def LeNet(input_shape):
  '''
  Creates and returns a LeNet based CNN model
  '''
  model = Sequential()
  model.add(Cropping2D(((70, 25), (0, 0)), input_shape=input_shape))
  model.add(Lambda(lambda x: x / 255.0 - 0.5))
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Conv2D(6, (5, 5), activation='relu'))
  model.add(MaxPooling2D())
  model.add(Flatten())
  model.add(Dense(120))
  model.add(Dense(84))
  model.add(Dense(1))
  return model

def NvidiaNet(input_shape):
  '''
  Creates and returns a CNN model based on the model described in
  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
  '''
  model = Sequential()
  model.add(Cropping2D(((70, 25), (0, 0)), input_shape=input_shape))
  model.add(Lambda(lambda x: x / 255.0 - 0.5))
  model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
  model.add(Dropout(0.25))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.25))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dropout(0.25))
  model.add(Dense(10))
  model.add(Dense(1))
  return model

def main():
  # all available data paths
  data_paths = ['data/', 'data2/']
  lines = read_logs(data_paths)
  # print first line for debugging
  print(lines[0])

  # angle correction for left and right camera images
  # this value is obtained empirically
  angle_correction = 0.07

  # read all data in to the memory
  # since data consumes too much memory and sometimes the code crashes
  # with an out of memory error, generator is used instead.
  #X_train, y_train = read_data(lines, angle_correction)
  # print info for debugging
  #print(X_train.shape, y_train.shape, X_train.dtype)

  batch_size = 32
  # samples are all image variations
  samples = generate_samples(lines)
  print('Total number of samples:', len(samples))
  # split all variations into two sets, train and validation sets
  train_samples, valid_samples = train_test_split(samples, test_size=0.2)
  train_generator = batch_generator(train_samples, angle_correction, batch_size=batch_size)
  valid_generator = batch_generator(valid_samples, angle_correction, batch_size=batch_size)

  # create model
  input_shape = (160, 320, 3)
  #model = LeNet(input_shape)
  model = NvidiaNet(input_shape)
  model.compile(loss='mse', optimizer='adam')

  # for saving only the best model, a keras model checkpoint is used
  checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
  #history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10, callbacks=[checkpointer])
  history = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size, epochs=10, validation_data=valid_generator, validation_steps=len(valid_samples)/batch_size, callbacks=[checkpointer])


  # visualize training history
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Valid'], loc='upper right')
  plt.savefig('loss.png')

  # visualize model
  # model visualization needs pydot and graphviz libraries
  # without them it gives error
  #plot_model(model, to_file='model.png', show_shapes=True)

if __name__ == "__main__":
    main()