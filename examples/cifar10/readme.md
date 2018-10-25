# Training HWGQ VGG-Net on CIFAR10

We will assume that you have HWGQ successfully compiled, and the installation is located at `HWGQ_ROOT`.

## Prepare Datasets

You will first need to download and convert the data format from the [CIFAR-10 website](http://www.cs.toronto.edu/~kriz/cifar.html). To do this, simply run the following commands:

    cd $HWGQ_ROOT
    ./data/cifar10/get_cifar10.sh
    ./examples/cifar10/create_cifar10.sh

For the training data, we use a padded version of CIFAR10, in which each image of 32x32 pixels is replicate padded by 4 pixels at each side to 40x40 pixels. You can download the lmdb by the following commands:

    cd $HWGQ_ROOT/examples/cifar10
    sh get_cifar10_pad_lmdb.sh


## Training HWGQ VGG-Net

The model is a small version of the popular VGG-Net. Start training by the following command:

    $HWGQ_ROOT/examples/cifar10/vgg-hwgq-3ne-clip-poly/
    sh train_vgg_cifar10.sh

The final results will have some noise, but the accuracy should be close to 92.0%~92.5%.
