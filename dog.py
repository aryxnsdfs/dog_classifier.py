import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ========== CONFIG ==========
DATA_DIR = r'c:\Users\aryan\Downloads\dog\train'
CSV_PATH = r'c:\Users\aryan\Downloads\dog\labels.csv'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

# ========== LOAD DATA ==========
df = pd.read_csv(CSV_PATH)
df['id'] = df['id'] + '.jpg'
df.columns = ['filename', 'class']

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class'], random_state=42)

# ========== IMAGE AUGMENTATION ==========
train_gen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    rotation_range=30,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2
)

val_gen = ImageDataGenerator(rescale=1./255)

train_da = train_gen.flow_from_dataframe(
    dataframe=train_df,
    directory=DATA_DIR,
    x_col='filename',
    y_col='class',
    class_mode='categorical',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_da = val_gen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_DIR,
    x_col='filename',
    y_col='class',
    class_mode='categorical',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ========== CLASS COUNT & WEIGHTING ==========
num_classes = len(train_da.class_indices)

# Get mapping from class name to index
class_to_index = train_da.class_indices

# Convert string classes to index values
indexed_y = train_df['class'].map(class_to_index)

# Compute class weights using numeric labels
from sklearn.utils.class_weight import compute_class_weight
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(indexed_y),
    y=indexed_y
)
class_weights_dict = dict(zip(np.unique(indexed_y), class_weights_array))


# ========== LOAD BASE MODEL ==========
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base.layers:
    layer.trainable = False

# ========== CUSTOM CLASSIFICATION HEAD ==========
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=predictions)

# ========== COMPILE & TRAIN BASE ==========
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

history = model.fit(
    train_da,
    validation_data=val_da,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# ========== FINE-TUNING ==========
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model.fit(
    train_da,
    validation_data=val_da,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights_dict
)

# ========== SAVE MODEL ==========
model.save('fine.h5')
