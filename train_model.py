#%% Imports

import napari
import numpy as np
from skimage import io 
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%% Open data

raw = []; mask = []
dirlist = sorted(Path('data_RBCs', 'train').glob('*.tif'))
for path in dirlist:
    if 'mask' not in path.name: 
        raw.append(io.imread(path))
    else:
        mask.append(io.imread(path))

raw = np.stack(raw)
mask = np.stack(mask)

#%% Prepare data 

# Normalize raw (0 to 1)
raw = (raw-np.min(raw))/(np.max(raw)-np.min(raw))

# Binarize mask (Boolean)
mask = mask > 0 

#%% Design the model

def double_conv_block(x, n_filters):

    x = layers.Conv2D(
        n_filters, 3, 
        padding = "same", 
        activation = "relu", 
        kernel_initializer = "he_normal"
        )(x)

    x = layers.Conv2D(
        n_filters, 3, 
        padding = "same", 
        activation = "relu", 
        kernel_initializer = "he_normal"
        )(x)
    
    return x

# -----------------------------------------------------------------------------

def downsample_block(x, n_filters):
   
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
   
    return f, p

# -----------------------------------------------------------------------------

def upsample_block(x, conv_features, n_filters):

    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    
    return x

# -----------------------------------------------------------------------------

def build_unet_model():

    # inputs
    inputs = layers.Input(shape=(128,128,1))
    
    # encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)
    
    # bottleneck
    bottleneck = double_conv_block(p4, 1024)
    
    # decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    
    # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    
    return unet_model

# -----------------------------------------------------------------------------

unet_model = build_unet_model()
# unet_model.summary()

# -----------------------------------------------------------------------------

unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics='accuracy')

# -----------------------------------------------------------------------------

results = unet_model.fit(
    raw, mask, 
    validation_split=0.33, 
    batch_size=32, 
    epochs=20, 
    )


#%% Display

# viewer = napari.Viewer()
# viewer.add_image(mask)






