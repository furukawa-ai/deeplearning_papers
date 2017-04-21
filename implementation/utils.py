"""
utility functions

170421 initial commit

* load_mnist_1D
* load_mnist_3D
* save_model_viz
* save_weights
* save_hist
* plot_hist
"""

def load_mnist_1D(categorical=True):
    '''
    load 1D mnist dataset w/ normalization + one-hot encoding
    '''
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1,784)
    X_train = X_train.astype('float32')/255.
    X_test = X_test.reshape(-1,784)
    X_test = X_test.astype('float32')/255.

    if categorical == True:
        from keras.utils import np_utils
        y_train = np_utils.to_categorical(y_train, 10)
        y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def load_mnist_3D():
    '''
    load 3D mnist dataset w/ normalization + one-hot encoding
    '''
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1,28,28,1)
    X_train = X_train.astype('float32')/255.
    X_test = X_test.reshape(-1,28,28,1)
    X_test = X_test.astype('float32')/255.

    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, y_train), (X_test, y_test)


def save_model_viz(run_id, model):
    '''
    save log_history + model_figure to 'model/'
    '''
    import os
    if not os.path.exists('model/'):
        os.makedirs('model/')

    from keras.utils import plot_model
    plot_model(model, to_file='model/'+run_id+'_vis.png',
               show_shapes=True, show_layer_names=True)


def save_weights(run_id, model):
    '''
    save learned weights to 'model/'
    '''
    import os
    if not os.path.exists('model/'):
        os.makedirs('model/')

    open('model/'+run_id+'.yaml', 'w').write(model.to_yaml())
    model.save_weights('model/'+run_id+'_weight.h5')


def save_hist(run_id, history):
    '''
    save log_history to 'log/'
    '''
    import os
    if not os.path.exists('log/'):
        os.makedirs('log/')

    import pandas as pd
    df = pd.DataFrame(history.history)
    df.to_csv('log/'+run_id+'_history.csv', index=False)


def plot_hist(run_id, val=True):
    '''
    plot trainig history
    '''
    import pandas as pd
    df = pd.read_csv('log/'+run_id+'_history.csv')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(df['acc'],'o-',label='accuracy')
    if val==True:
        plt.plot(df['val_acc'],'o-',label='val_acc')
    plt.title(run_id)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(min(df['acc'])*0.8, 1.)
    plt.legend(loc='lower right')
    fig.savefig('log/'+run_id+'_acc.png')

    fig = plt.figure()
    plt.plot(df['loss'],'o-',label='loss',)
    if val==True:
        plt.plot(df['val_loss'],'o-',label='val_loss')
    plt.title(run_id)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, max(df['loss'])*1.2)
    plt.legend(loc='upper right')
    fig.savefig('log/'+run_id+'_loss.png')
