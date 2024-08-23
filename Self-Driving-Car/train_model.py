import numpy as np
from sklearn.metrics import r2_score
from keras.layers import Dense, Flatten, Conv2D, Lambda
from keras.layers import MaxPooling2D, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.optimizers import Adam
import keras.backend as K
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Function to calculate R2 score at each epoch 
class R2ScoreCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_x, val_y = self.validation_data
        predictions = self.model.predict(val_x).flatten()
        r2 = r2_score(val_y, predictions)
        print(f'\nEpoch {epoch + 1} R² score: {r2:.4f}')

# CNN Model
def cnn_model(image_x, image_y):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(image_x, image_y, 1)))

    # Six Convolutional Layers with Batch Normalization and LeakyRelu as activation function
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(128, (4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    
    #Flatten layer before going towards fully connected layer
    model.add(Flatten())

    # Three Fully Connected layers 
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.1))

    # Output Layer
    model.add(Dense(1, activation='linear'))
    
    # Compiling the model 
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    filepath = "models/Autopilot_new_7.keras"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    return model, callbacks_list

# Reading the binary features and labels file into the model
def loadFromPickle():
    with open("features", "rb") as f:
        features = np.array(pickle.load(f))
    with open("labels", "rb") as f:
        labels = np.array(pickle.load(f))

    return features, labels

# Model Training
def main():
    features, labels = loadFromPickle()
    features, labels = shuffle(features, labels)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                        test_size=0.3)
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    
    model, callbacks_list = cnn_model(100, 100)

    r2_callback = R2ScoreCallback(validation_data=(test_x, test_y))
    callbacks_list.append(r2_callback)
    
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=30, batch_size=128,
              callbacks=callbacks_list)
    model.summary()

    # Saving Model
    model.save('models/Autopilot_new_7.keras')

main()
K.clear_session();
