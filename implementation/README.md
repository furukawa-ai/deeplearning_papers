## Installation of Keras

```
$ pip install -U tensorflow keras h5py
```

## Installation for Network Visualization

ubuntu

```
$ sudo apt-get install graphviz libgraphviz-dev pkg-config
$ pip install -U pygraphviz pydot pydot_ng
```

mac

```
$ brew install graphviz
$ pip install pygraphviz pydot pydot_ng
```

## Examples of Network Visualization

```
from keras.utils import plot_model
plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)
# if jupyter environment
from IPython.display import Image
Image("model_lstm.png", width=400)
```
