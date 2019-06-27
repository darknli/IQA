from tensorflow.python.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Conv2D, Flatten, Softmax, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy

import os
# import warnings
# warnings.filterwarnings('ignore')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


classes = ['bad', 'good']

early_stopper = EarlyStopping(patience=10)

class SiameseModel:
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
        output = Dense(1, activation='sigmoid')(x)
        # output = SiameseLossLayer(self.batch_size, self.num_distort, self.num_level)(x)
        model = Model(inputs=base_model.input, outputs=output)
        self.name = model_name
        return model

    def set_loss_param(self, batch_size,  num_distort, num_level):
        self.batch_size = batch_size
        self.num_distort = num_distort
        self.num_level = num_level

    def freeze_all_but_top(self, loss_type="siamese"):
        """Used to train just the top layers of the model."""
        for layer in self.model.layers[:-2]:
            layer.trainable = False
        for layer in self.model.layers[-2:]:
            layer.trainable = True

        if loss_type == "siamese":
            loss = self.siamese_loss
            try:
                test = self.batch_size
            except:
                raise AttributeError(
                    'It needs to execute member function \'set_loss_param\' before run this function'
                )
        elif loss_type == "mse":
            loss = "mean_squared_error"
        else:
            raise ValueError('Loss function must be in (siamese, mse)')

        self.model.compile(optimizer='adam', loss=loss)

    def freeze_all_but_mid_and_top(self, num_layer=172, loss_type="siamese"):
        """After we fine-tune the dense layers, train deeper."""
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        for layer in self.model.layers[:num_layer]:
            layer.trainable = False
        for layer in self.model.layers[num_layer:]:
            layer.trainable = True

        if loss_type == "siamese":
            loss = self.siamese_loss
        elif loss_type == "mse":
            loss = "mean_squared_error"
        else:
            raise ValueError('loss function must be in (siamese, mse)')

        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss=loss,
        )\

    def fit(self, nb_epoch, train, val, save_model_dir):

        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(save_model_dir, '{val_loss:.5f}-%s.h5' % self.name),
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss'
        )

        steps_per_train, train_data = train
        steps_per_val, val_data = val
        self.model.fit_generator(
            train_data,
            validation_data=val_data,
            validation_steps=steps_per_val,
            steps_per_epoch=steps_per_train,
            epochs=nb_epoch,
            workers=8,
            verbose=1,
            use_multiprocessing=True,
            callbacks=[checkpointer]
        )

    def summary(self):
        self.model.summary()

    def siamese_loss(self, y_true, y_pred):
        self.dis = []
        loss = 0
        margin = 0.2
        sum = 0
        for batch in range(self.batch_size):
            batch_step = batch*self.num_level*self.num_level
            for distort in range(self.num_distort):
                distort_step = distort*self.num_level
                step = batch_step+distort_step
                for i in range(self.num_level - 1):
                    for j in range(i+1, self.num_level):
                        # loss += K.maximum(y_pred[batch * distort + i, :] - y_pred[batch * distort + j, :], 0)
                        # loss += 1/(K.maximum(y_pred[batch * distort + i, :] - y_pred[batch * distort + j, :], 0)+1)
                        diff = y_pred[step + j, :] - y_pred[step + i, :]+margin
                        loss += K.maximum(diff, 0)
                        sum += 1
        return K.sum(loss)

    def load_model(self, weights):
        self.model.load_weights(weights)
        print("load model weights %s"%weights)

    def predict(self, path, is_hundred=True):
        coefficient = 1
        if is_hundred:
            coefficient = 100
        if os.path.isfile(path):
            img = self.process_image(path)
            score = self.model.predict(img)[0][0]
            return {path: coefficient*score}
        elif os.path.exists(path):
            result = {}
            for img_name in glob(os.path.join(path, '*')):
                try:
                    img = self.process_image(img_name)
                    score = self.model.predict(img)[0][0]
                except BaseException:
                    continue
                result[img_name] = coefficient*score
            return result

    def process_image(self, image):
        """Given an image, process it and return the array."""
        img = cv2.imread(image)
        return np.expand_dims(img, axis=0)

class SiameseLossLayer(Layer):
    def __init__(self, batch_size,  num_distort, num_level, **kwargs):
        self.is_placeholder = True
        self.batch_size = batch_size
        self.num_distort = num_distort
        self.num_level = num_level
        super(SiameseLossLayer, self).__init__(**kwargs)

    def loss(self, inputs):
        loss = 0
        for batch in range(self.batch_size):
            for distort in range(self.num_distort):
                for i in range(self.num_level-1):
                    for j in range(i, self.num_level):
                        loss += K.maximum(inputs[batch*distort+i, :]-inputs[batch*distort+j, :], 0)
        return K.sum(loss)

    def call(self, inputs):
        loss = self.loss(inputs)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return inputs
