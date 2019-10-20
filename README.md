# img2vec-keras
Image to dense vector embedding. This library uses the ResNet50 model in TensorFlow Keras, pre-trained on Imagenet, to generate image embeddings via https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer. Basically a clone of https://github.com/christiansafka/img2vec for TensorFlow Keras users. 

# Install
First you need TensorFlow backend installed. `img2vec_keras` uses the `keras` module shipped with `tensorflow`. Then,

```pip install git+git://github.com/jaredwinick/img2vec-keras.git```

# Usage

```python
from img2vec_keras import Img2Vec
from PIL import Image

img2vec = Img2Vec()
x = img2vec.get_vec('/path/to/image/dog1.jpg')

dog = Image.open('path/to/image/dog1.jpg')
x = Img2vec(img=dog)
```

Alternatively if you have already a pipeline in place to extract images into
`nd.array` type you can use the sklearn-style `.transform()` method.
```python
from PIL import Image

img_files = ["/path/to/image/dog1.jpg", "/path/to/image/dog2.jpg"]
dogs = [Image.open(f) for f in img_files]
dogs_array = [np.expand_dims(image.img_to_array(v.resize((224, 224)).convert("RGB")), 0)
              for v in dogs]
img2vec = Img2Vec()
img2vec.transform(dogs_array)
```

# Examples

[Basic example with cosine similarity of vectors](https://github.com/jaredwinick/img2vec-keras/blob/master/examples/similarity.py)

[Colab notebook using t-SNE to visualize image vectors](https://colab.research.google.com/drive/14OvmH6KvoQJ41jb6QRL3FgwI61vq-UAJ)
