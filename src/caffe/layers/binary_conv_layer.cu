#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BinWeightForward(const int n, const int weight_dim, const Dtype* weight, 
          const Dtype* alpha, Dtype* binary_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index/weight_dim;
    Dtype binary_code = (weight[index]>=0) ? 1:-1; 
    binary_weight[index] = binary_code*alpha[i];
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BinaryConvolutionParameter binary_conv_param = this->layer_param_.binary_convolution_param();
  bool use_alpha = binary_conv_param.use_alpha();
  bool use_binarization = binary_conv_param.use_binarization();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  binary_weights_.ReshapeLike(*this->blobs_[0]);
  caffe_gpu_memcpy(binary_weights_.count()*sizeof(Dtype),weight,binary_weights_.mutable_gpu_data());
  alphas_.Reshape(this->num_output_,1,1,1);
  caffe_gpu_set(this->num_output_,Dtype(1),alphas_.mutable_gpu_data());
  weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  caffe_gpu_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_gpu_data());
  const int nthreads = this->num_output_*weight_dim;

  // binarize the weights
  if (use_binarization) {
    // compute alpha if needed
    if (use_alpha) {
      caffe_gpu_abs(this->blobs_[0]->count(),weight,binary_weights_.mutable_gpu_diff());
      const Dtype* abs_weight = binary_weights_.gpu_diff();   
      caffe_gpu_gemv<Dtype>(CblasNoTrans, this->num_output_, weight_dim,
          1. / weight_dim, abs_weight, weight_sum_multiplier_.gpu_data(), 0.,
          alphas_.mutable_gpu_data());
    }
    BinWeightForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, weight_dim, weight, alphas_.gpu_data(), binary_weights_.mutable_gpu_data());
  }

  const Dtype* binary_weights = binary_weights_.gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, binary_weights,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binary_weights = binary_weights_.gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, binary_weights,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryConvolutionLayer);

}  // namespace caffe
