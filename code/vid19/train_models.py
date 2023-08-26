import numpy as np
import tensorflow as tf
from mnist_loader import load_training_data, load_validation_data,load_test_data,categorical_to_sparse
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

training_data=load_training_data()
x,y=training_data
training_data=(x,y)
validation_data=load_validation_data()

datagen=tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
    )

datagen.fit(x)

def train_model(model_name):
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(20,5,activation='relu',input_shape=(28,28,1)))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Conv2D(20,5,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(200,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10,activation='sigmoid'))
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy'])
    checkpoint=tf.keras.callbacks.ModelCheckpoint(model_name,
                                                  monitor='val_accuracy',
                                                  save_best_only=True,
                                                  mode='max')
    model.summary()
    model.fit(datagen.flow(x,y,batch_size=10),epochs=50,
              validation_data=validation_data,callbacks=[checkpoint])
    return model
models=[]

for i in range(5):
    models.append(train_model(f"model{i+1}"))
    

    

    

