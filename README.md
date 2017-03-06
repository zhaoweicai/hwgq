## Deep Learning with Low Precision by Half-wave Gaussian Quantization

by Zhaowei Cai, Xiaodong He, Jian Sun and Nuno Vasconcelos

This implementation is written by Zhaowei Cai at UC San Diego.

### Introduction

HWGQ-Net is a low-precision neural network with 1-bit binary weights and 2-bit quantized activations. It can be applied to many popular network architectures, including AlexNet, ResNet, GoogLeNet, VggNet, and achieves closer performance to the corresponding full-precision networks than previously available low-precision networks. Theorectically, HWGQ-Net has ~32x memory and ~32x convoluational computation savings, suggesting that it can be very useful for the deployment of state-of-the-art neural networks in real world applications. More details can be found in our [paper](https://arxiv.org/abs/1702.00953).

### Results

<p align="left">
<img src="http://www.svcl.ucsd.edu/projects/hwgq/hwgq_results.png" alt="HWGQ results" width="450px">
</p>

### Citation

If you use our code, please cite our paper:

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

0. These models are compatible with the provided training\finetuning scripts. The weights of the models here are not binariazed yet. Binarization happens during running.
	- [AlexNet_HWGQ] (http://www.svcl.ucsd.edu/projects/hwgq/AlexNet_HWGQ.caffemodel)
	- [ResNet-101] (http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50)
	- [ResNet-152] (http://ethereon.github.io/netscope/#/gist/d38f3e6091952b45198b)

0. Model files:
	- ~~MSR download: [link] (http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip)~~
	- OneDrive download: [link](https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777)

If you encounter any issue when using our code, please let me know.
