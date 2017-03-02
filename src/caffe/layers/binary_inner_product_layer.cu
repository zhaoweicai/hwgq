#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/binary_inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void IPBinWeightForward(const int n, const int weight_dim, const Dtype* weight, 
          const Dtype* alpha, Dtype* binary_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index/weight_dim;
    Dtype binary_code = (weight[index]>=0) ? 1:-1; 
    binary_weight[index] = binary_code*alpha[i];
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  BinaryInnerProductParameter binary_inner_product_param = this->layer_param_.binary_inner_product_param();
  bool use_alpha = binary_inner_product_param.use_alpha();
  bool use_binarization = binary_inner_product_param.use_binarization();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  binary_weights_.ReshapeLike(*this->blobs_[0]);
  caffe_copy(binary_weights_.count(),weight,binary_weights_.mutable_gpu_data());
  alphas_.Reshape(num_output,1,1,1);
  caffe_gpu_set(num_output,Dtype(1),alphas_.mutable_gpu_data());
  weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  caffe_gpu_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_gpu_data());
  const int nthreads = num_output*weight_dim;

  // binarize the weights
  if (use_binarization) {
    // compute alpha if needed
    if (use_alpha) {
      caffe_gpu_abs(this->blobs_[0]->count(),weight,binary_weights_.mutable_gpu_diff());
      const Dtype* abs_weight = binary_weights_.gpu_diff();    
      caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output, weight_dim,
          1. / weight_dim, abs_weight, weight_sum_multiplier_.gpu_data(), 0.,
          alphas_.mutable_gpu_data());
    }
    IPBinWeightForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, weight_dim, weight, alphas_.gpu_data(), binary_weights_.mutable_gpu_data());
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* binary_weights = binary_weights_.gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         binary_weights, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, binary_weights, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void BinaryInnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binary_weights = binary_weights_.gpu_data();
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, binary_weights,
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, binary_weights,
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BinaryInnerProductLayer);

}  // namespace caffe
