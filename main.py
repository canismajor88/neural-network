import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import pandas as pd
import numpy as np

df = pd.read_csv(r"aug_train.csv")
#skewed too many males
del df['gender']
del df['enrollee_id']
del df['city']
df = pd.get_dummies(df)
df.to_csv("cleanedCsv.csv")
train = df.sample(frac=.7)
df = df.drop(train.index)
test = df.sample(frac=.5)
df = df.drop(test.index)
validation = df

target = train.pop('target')
trainTarget = tf.keras.utils.to_categorical(target)
Target=test.pop('target')
testTarget = tf.keras.utils.to_categorical(Target)
Target=validation.pop('target')
validationTarget=tf.keras.utils.to_categorical(Target)
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.tanh),
        tf.keras.layers.Dense(30, activation=tf.nn.tanh),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=['categorical_accuracy'])
    return model
checkpoint_path="mymodelcp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model = get_compiled_model()
model.fit(train, trainTarget, epochs=20, validation_data=(test,testTarget),
         batch_size=10,callbacks=[cp_callback])
model.evaluate(validation, validationTarget, batch_size=1)