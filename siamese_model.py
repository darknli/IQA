from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.engine.base_layer import Layer
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

class FRIAQModel:
    def __init__(self, model_name, weights="imagenet"):
        self.model = self.get_model(model_name, weights)

    def get_model(self, model_name, weights='imagenet'):
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
            base_model = InceptionResNetV2(weights=weights, include_top=False)
        else:
            raise KeyError('Unknown network.')
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(1, activation='relu')(x)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    def freeze_all_but_top(self, model):
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

    def freeze_all_but_mid_and_top(self, model, num_layer=172):
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


class SiameseLossLayer(Layer):
    def __init__(self, batch_size,  num_distort, num_level, **kwargs):
        self.is_placeholder = True
        self.batch_size = batch_size
        self.num_distort = num_distort
        self.num_level = num_level
        super(SiameseLossLayer, self).__init__(**kwargs)

    def loss(self, inputs):

        self.dis = []
        loss = 0
        for batch in range(self.batch_size):
            for distort in range(self.num_distort):
                for i in range(self.num_level-1):
                    for j in range(i, self.num_level):
                        loss += K.maximum(inputs[batch*distort+i, :]-inputs[batch*distort+j:])
        return loss

    def call(self, inputs):
        loss = self.loss(inputs)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return inputs
