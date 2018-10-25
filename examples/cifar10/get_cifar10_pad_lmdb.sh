echo "Downloading CIFAR10 data..."

wget -c http://www.svcl.ucsd.edu/projects/hwgq/cifar10_pad_train_lmdb.zip

echo "Unzipping..."

unzip cifar10_pad_train_lmdb.zip && rm -f cifar10_pad_train_lmdb.zip

echo "Done."
