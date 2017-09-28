import os.path as op

from tensorflow.examples.tutorials import mnist

from datasets import DataSet

mnist_dir = op.join(op.expanduser('~'), 'Data', 'tf_MNIST')
mnist_data = mnist.input_data.read_data_sets(mnist_dir, one_hot=True)
print(type(mnist_data))
md = mnist_data.train


def get_dataset(_d, name: str) -> DataSet:
    _x = _d.images
    n_samples = (_x.shape[0])
    name = "MNIST-" + name
    return DataSet(_x.reshape(n_samples, 28, 28), _d.labels, name=name)


[train,
 test,
 validation] = map(get_dataset,
                   (mnist_data.train,
                    mnist_data.test,
                    mnist_data.validation),
                   ("train", "test", "validation"))

