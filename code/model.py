from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
import time
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf


# Model/Training parameters
NUM_EPOCH = 10
BATCH_SIZE = 16
H, W, CH = 160, 320, 3
LR = 1e-4
L2_REG_SCALE = 0.


def gen(data, labels, batch_size):
    """
    Batch generator
    Arguments:
        data: List of strings containing the path of image files
        labels: List of steering angles
        batch_size: Size of the batch to generate
    Yields:
        A tuple (X_batch, y_batch), where:
            X_batch: Batch of images, a tensor of shape (batch_size, H, W, CH)
            y_batch: Batch of steering angles
    """
    start = 0
    end = start + batch_size
    n = data.shape[0]

    while True:
        # Read image data into memory as-needed
        image_files  = data[start:end]
        images = []
        for image_file in image_files:
            # Resize image, create numpy array representation
            image = Image.open(image_file).convert('RGB')
            image = image.resize((W, H), Image.ANTIALIAS)
            image = np.asarray(image, dtype='float32')
            images.append(image)
        images = np.array(images, dtype='float32')

        X_batch = images
        y_batch = labels[start:end]
        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size

        yield (X_batch, y_batch)


def get_model():
    
    ch, row, col = CH, H, W  # camera format

    model = Sequential()

    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

    # trim image to only see section with road
    model.add(Cropping2D(cropping=((70,25),(0,0))))           

    #layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 2- Convolution, no of filters- 36, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(Activation('elu'))

    #layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
    model.add(Convolution2D(64,3,3))
    model.add(Activation('elu'))

    #flatten image from 2D to side by side
    model.add(Flatten())

    #layer 6- fully connected layer 1
    model.add(Dense(100))
    model.add(Activation('elu'))

    #Adding a dropout layer to avoid overfitting. Here we are have given the dropout rate as 25% after first fully connected layer
    model.add(Dropout(0.25))

    #layer 7- fully connected layer 1
    model.add(Dense(50))
    model.add(Activation('elu'))


    #layer 8- fully connected layer 1
    model.add(Dense(10))
    model.add(Activation('elu'))

    #layer 9- fully connected layer 1
    model.add(Dense(1)) #here the final layer will contain one value as this is a regression problem and not classification


    # the output is the steering angle
    # using mean squared error loss function is the right choice for this regression problem
    # adam optimizer is used here
    model.compile(loss='mse',optimizer='adam')

    return model


def train_model():
    # Load driving data
    with open('driving_data_prep_normal.p', mode='rb') as f:
        driving_data = pickle.load(f)

    data, labels = driving_data['images'], driving_data['labels']
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=0)

    # Get model
    model = get_model()
    model.summary()

    # Visualize model and save it to disk
    plot_model(model, to_file='model.png')
    print('Saved model visualization at model.png')

    # Instantiate generators
    train_gen = gen(X_train, y_train, BATCH_SIZE)
    val_gen = gen(X_val, y_val, BATCH_SIZE)

    train_start_time = time.time()

    # Train model
    h = model.fit_generator(generator=train_gen, samples_per_epoch=X_train.shape[0], nb_epoch=NUM_EPOCH, validation_data=val_gen, nb_val_samples=X_val.shape[0])
    history = h.history

    total_time = time.time() - train_start_time
    print('Total training time: %.2f sec (%.2f min)' % (total_time, total_time/60))

    # Save model architecture to model.json, model weights to model.h5
    json_string = model.to_json()
    with open('model.json', 'w') as f:
        f.write(json_string)
    model.save_weights('model.h5')

    # Save training history
    with open('train_hist.p', 'wb') as f:
        pickle.dump(history, f)

    print('Model saved in model.json/h5, history saved in train_hist.p')


if __name__ == '__main__':
    print('Training comma.ai model')
    train_model()
    print('DONE: Training comma.ai model')
