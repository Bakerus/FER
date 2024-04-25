
# import common python librairies
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import matplotlib.pyplot as plt

# import Deep learning librairies
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, GlobalAveragePooling2D, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator()
validation_data_gen = ImageDataGenerator()

# Preprocess all train images
train_set = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=128,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle= True)

# # Preprocess all test images
# test_set = validation_data_gen.flow_from_directory(
#         'data/test',
#         target_size=(48, 48),
#         batch_size=128,
#         color_mode="grayscale",
#         class_mode='categorical',
#         shuffle= False)


# create model structure
number_of_classes = 7
emotion_model = Sequential()

# 1st CNN layer
emotion_model.add(Conv2D(64, (3, 3), padding= 'same', input_shape=(48, 48, 1) ))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size = (2,2)))
emotion_model.add(Dropout(0.25))

# 2nd CNN layer
emotion_model.add(Conv2D(128, (5, 5), padding= 'same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size = (2,2)))
emotion_model.add(Dropout(0.25))

# 3rd CNN layer
emotion_model.add(Conv2D(512, (3, 3), padding= 'same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size = (2,2)))
emotion_model.add(Dropout(0.25))

# 4th CNN layer
emotion_model.add(Conv2D(512, (3, 3), padding= 'same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size = (2,2)))
emotion_model.add(Dropout(0.25))

# convolved matrix flattening 
emotion_model.add(Flatten())

# Fully connected 1st layer
emotion_model.add(Dense(256))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))

# Fully connected 2nd layer
emotion_model.add(Dense(512))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))


emotion_model.add(Dense(number_of_classes, activation='softmax'))

# cv2.ocl.setUseOpenCL(False)
# model compilation
emotion_model.compile(loss='categorical_crossentropy', optimizer= Adam(learning_rate=0.001), metrics=['accuracy'])


checkpoint = ModelCheckpoint("model.keras", monitor='val_acc', verbose=1 , save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor= 'val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)

redure_learningrate = ReduceLROnPlateau(monitor= 'val_loss',
                               factor=0.2,
                               patience=3,
                               verbose=1,
                               min_delta=0.0001)

callbacks_list = [early_stopping, checkpoint, redure_learningrate]


# Train the neural network/model
emotion_model_info = emotion_model.fit(
        train_set,
        steps_per_epoch= train_set.n // train_set.batch_size,
        epochs=60,
        validation_data= None,
        callbacks=callbacks_list
        # validation_steps=7178 // 64
        )



# Courbe d'apprentissage
acc = emotion_model_info.history['accuracy']
#val_acc = history.history['val_accuracy']
loss = emotion_model_info.history['loss']
#val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
#plt.ylim([min(plt.ylim()),1])
#plt.title('Training and Validation Accuracy')
plt.title('Training Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
#plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
#plt.ylim([0,1.0])
#plt.title('Training and Validation Loss')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.show()



# save model structure in jason file
emotion_model.save('emotion_model.keras')
print("The training is over")
# model_json = emotion_model.to_json()j
# with open("emotion_model.json", "w") as json_file:
#     json_file.write(model_json)

# save trained model weight in .h5 file
# emotion_model.save_weights('emotion_model.h5')