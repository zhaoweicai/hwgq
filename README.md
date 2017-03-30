## Deep Learning with Low Precision by Half-wave Gaussian Quantization

by [Zhaowei Cai](https://sites.google.com/site/zhaoweicai1989/), [Xiaodong He](https://www.microsoft.com/en-us/research/people/xiaohe/?from=http%3A%2F%2Fresearch.microsoft.com%2F~xiaohe), [Jian Sun](http://www.jiansun.org/) and [Nuno Vasconcelos](http://www.svcl.ucsd.edu/~nuno/).

This implementation is written by Zhaowei Cai at UC San Diego.

### Introduction

HWGQ-Net is a low-precision neural network with 1-bit binary weights and 2-bit quantized activations. It can be applied to many popular network architectures, including AlexNet, ResNet, GoogLeNet, VggNet, and achieves closer performance to the corresponding full-precision networks than previously available low-precision networks. Theorectically, HWGQ-Net has ~32x memory and ~32x convoluational computation savings, suggesting that it can be very useful for the deployment of state-of-the-art neural networks in real world applications. More details can be found in our [paper](https://arxiv.org/abs/1702.00953).

### Results

<p align="left">
<img src="http://www.svcl.ucsd.edu/projects/hwgq/hwgq_results.png" alt="HWGQ results" width="450px">
</p>

### Citation

If you use our code/model/data, please cite our paper:

    @inproceedings{cai17hwgq,
      author = {Zhaowei Cai and Xiaodong He and Jian Sun and Nuno Vasconcelos},
      Title = {Deep Learning with Low Precision by Half-wave Gaussian Quantization},
      booktitle = {CVPR},
      Year  = {2017}
    }

### Installation

1. Clone the HWGQ repository, and we'll call the directory that you cloned HWGQ into `HWGQ_ROOT`
    ```Shell
    git clone https://github.com/zhaoweicai/hwgq.git
    ```
  
2. Build HWGQ
    ```Shell
    cd $HWGQ_ROOT/
    # Follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j 16
    ```

3. Set up ILSVRC2012 dataset following [Caffe ImageNet instruction](https://github.com/BVLC/caffe/tree/master/examples/imagenet).

### Training HWGQ

You can start training HWGQ-Net. Take AlexNet for example. 
```Shell
cd $HWGQ_ROOT/examples/imagenet/alex-hwgq-3ne-clip-poly-320k/
sh train_alexnet_imagenet.sh
```
    
The other network architectures are available, including ResNet-18, ResNet-34, ResNet-50, GoogLeNet and VggNet. To train deeper networks, you need multi-GPU training. If multi-GPU training is not available to you, consider to change the batch size and the training iterations accordingly. You can get very close performance as in our paper.

### Running Ablation Experiments

Most of the ablation experiments in the paper can be reproduced. The training scripts for running `BW+sign` (`$HWGQ_ROOT/examples/imagenet/alex-sign-step-160k`) in Table 2 and 2-bit non-uniform/uniform `BW+HWGQ` (`$HWGQ_ROOT/examples/imagenet/alex-hwgq-2n-clip-step-160k` and `$HWGQ_ROOT/examples/imagenet/alex-hwgq-3ne-clip-step-160k`) in Table 4 of AlexNet are provided here for reproduction. 

### Models

0. These models are compatible with the provided training/finetuning scripts. The weights of the models here are not binarized yet. Binarization happens during running.
	- [AlexNet_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/AlexNet_HWGQ.caffemodel)
	- [ResNet18_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/ResNet18_HWGQ.caffemodel)
	- [ResNet34_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/ResNet34_HWGQ.caffemodel)
	- [ResNet50_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/ResNet50_HWGQ.caffemodel)
	- [GoogleNet_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/GoogleNet_HWGQ.caffemodel)
	- [VggNet_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/VggNet_HWGQ.caffemodel)
	- [DarkNet19_HWGQ](http://www.svcl.ucsd.edu/projects/hwgq/DarkNet19_HWGQ.caffemodel)

0. The weights of the models here are already binarized, and are compatible with standard convolutions. The deploy_bw.prototxt can be found in the corresponding folders. These models can be used for deployment.
	- [AlexNet_HWGQ_BW](http://www.svcl.ucsd.edu/projects/hwgq/AlexNet_HWGQ_BW.caffemodel)
	- [ResNet18_HWGQ_BW](http://www.svcl.ucsd.edu/projects/hwgq/ResNet18_HWGQ_BW.caffemodel)
	- [VggNet_HWGQ_BW](http://www.svcl.ucsd.edu/projects/hwgq/VggNet_HWGQ_BW.caffemodel)

If you encounter any issue when using our code/model, please let me know.
