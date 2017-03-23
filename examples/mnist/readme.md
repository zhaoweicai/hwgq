# Training a HWGQ LeNet on MNIST with Caffe

Adapted from the Caffe MNIST tutorial, where the base LeNet has been modified to use binary weights and 2-bit uniform HWGQ activations.

We will assume that you have Caffe successfully compiled. If not, please refer to the [Installation page](/installation.html). In this tutorial, we will assume that your Caffe installation is located at `CAFFE_ROOT`.

## Prepare Datasets

You will first need to download and convert the data format from the MNIST website. To do this, simply run the following commands:

    cd $CAFFE_ROOT
    ./data/mnist/get_mnist.sh
    ./examples/mnist/create_mnist.sh

If it complains that `wget` or `gunzip` are not installed, you need to install them respectively. After running the script there should be two datasets, `mnist_train_lmdb`, and `mnist_test_lmdb`.

## Train HWGQ LeNet

Once the datasets are ready, run the following to train a HWGQ LeNet:
    
    cd $CAFFE_ROOT
    ./examples/mnist/hwgq-lenet-train.sh
    
On a GTX 1070 GPU (+cuDNN) this finishes in 30 seconds and reaches 99% accuracy.
