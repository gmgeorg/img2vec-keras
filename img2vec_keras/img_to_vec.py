#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Text, Optional

from tensorflow import keras

from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np


_IMAGE_NET_TARGET_SIZE = (224, 224)


class Img2Vec(object):

    def __init__(self, layer_name: Text = "avg_pool"):
        
        model = resnet50.ResNet50(weights='imagenet')
        self._layer_name = layer_name
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(self._layer_name).output)

    def get_vec(self, image_path: Optional[Text] = None, img=None) -> np.ndarray:
        """ Gets a vector embedding from an image.

        :param image_path: path to image on filesystem
        :param img: image object (Pillow). If image_path is not provided, then
          img must be provided.

        :returns: numpy ndarray
        """
        if img is None:
            img = image.load_img(image_path, 
                                 target_size=_IMAGE_NET_TARGET_SIZE)
        else:
            img = img.resize(_IMAGE_NET_TARGET_SIZE)
            if img.mode != "RGB":
                # Convert it to 3-channel image.
                img = img.convert("RGB")

        return self.transform(np.expand_dims(image.img_to_array(img), axis=0))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """ Gets a vector embedding from an image array.

        The .transform() method follows sklearn-style and can therefore be used
        in sklearn Pipeline.
        
        :param X: images encoded as a 4D numpy array, with axis=0 being the
          dimension enumerating different images, and the remaining dimensions
          encoding (224, 224, 3), where (224, 224) are the width/height of
          array (must be 224,224) and 3 are the 3 RGB channels.

        :returns: The embedding of the image as numpy ndarray
        """
        n_images = X.shape[0]
        intermediate_output = self.intermediate_layer_model.predict(
            resnet50.preprocess_input(X))
        
        embeddings = intermediate_output[0]
        return embeddings.reshape((n_images, int(len(embeddings) / n_images)))


if __name__ == "main":
     pass    