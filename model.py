from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Conv2D, Flatten, Softmax, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy

import os
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


classes = ['bad', 'good']

early_stopper = EarlyStopping(patience=10)

def get_model(model_name, weights='imagenet'):
    # create the base pre-trained model
    # inputs = Input(shape=inputs_shape+(1,))
    if model_name == 'InceptionV3':
        from tensorflow.python.keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(weights=weights, include_top=False)
    elif model_name == 'NASNetLarge':
        from tensorflow.python.keras.applications.nasnet import NASNetLarge
        base_model = NASNetLarge(weights=weights, include_top=False)
    elif model_name == 'DenseNet201':
        from tensorflow.python.keras.applications.densenet import DenseNet201
        base_model = DenseNet201(weights=weights, include_top=False)
    elif model_name == 'Xception':
        from tensorflow.python.keras.applications.xception import Xception
        base_model = Xception(weights=weights, include_top=False)
    elif model_name == 'VGG19':
        from tensorflow.python.keras.applications.vgg19 import VGG19
        base_model = VGG19(weights=weights, include_top=False)
    elif model_name == 'NASNetMobile':
        from tensorflow.python.keras.applications.nasnet import NASNetMobile
        base_model = NASNetMobile(weights=weights, include_top=False)
    elif model_name == 'MobileNetV2':
        from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
        base_model = MobileNetV2(weights=weights, include_top=False)
    elif model_name == 'ResNet50':
        from tensorflow.python.keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights=weights, include_top=False)
    elif model_name == 'InceptionResNetV2':
        from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(weights=weights, include_top=False, )

    else:
        raise KeyError('Unknown network.')
    x = base_model.output
    # x = Conv2D(1024, (3, 3), activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(2, (8, 8))(x)
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(len(classes), activation='softmax')(x)
    # x = Flatten()(x)
    # predictions = Softmax()(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def freeze_all_but_top(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:-2]:
        layer.trainable = False
    for layer in model.layers[-2:]:
        layer.trainable = True
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def freeze_all_but_mid_and_top(model, num_layer=172):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:num_layer]:
        layer.trainable = False
    for layer in model.layers[num_layer:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model
