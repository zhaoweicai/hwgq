#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_binary_conv_layer.hpp"

namespace caffe {

__global__ void sync_binary_conv_groups() { }

template <typename Dtype>
__global__ void BinWeightCudnnForward(const int n, const int weight_dim, const Dtype* weight, 
          const Dtype* alpha, Dtype* binary_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index/weight_dim;
    Dtype binary_code = (weight[index]>=0) ? 1:-1; 
    binary_weight[index] = binary_code*alpha[i];
  }
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BinaryConvolutionParameter binary_conv_param = this->layer_param_.binary_convolution_param();
  bool use_alpha = binary_conv_param.use_alpha();
  bool use_binarization = binary_conv_param.use_binarization();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  this->weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  caffe_gpu_set(this->weight_sum_multiplier_.count(),Dtype(1),
                this->weight_sum_multiplier_.mutable_gpu_data());
  this->binary_weights_.ReshapeLike(*this->blobs_[0]);
  caffe_gpu_memcpy(this->binary_weights_.count() * sizeof(Dtype),weight,
                   this->binary_weights_.mutable_gpu_data());
  this->alphas_.Reshape(this->num_output_,1,1,1);
  caffe_gpu_set(this->num_output_,Dtype(1),this->alphas_.mutable_gpu_data());
  const int nthreads = this->num_output_*weight_dim;

  // binarize the weights
  if (use_binarization) {
    // compute alpha if needed
    if (use_alpha) {
      caffe_gpu_abs(this->blobs_[0]->count(),weight,this->binary_weights_.mutable_gpu_diff());
      const Dtype* abs_weight = this->binary_weights_.gpu_diff();
      caffe_gpu_gemv<Dtype>(CblasNoTrans, this->num_output_, weight_dim,
          1. / weight_dim, abs_weight, this->weight_sum_multiplier_.gpu_data(), 0.,
          this->alphas_.mutable_gpu_data());
    }
    BinWeightCudnnForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
        nthreads, weight_dim, weight, this->alphas_.gpu_data(), this->binary_weights_.mutable_gpu_data());
  }

  const Dtype* binary_weight = this->binary_weights_.gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, binary_weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));

      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_binary_conv_groups<<<1, 1>>>();
  }
}

template <typename Dtype>
void CuDNNBinaryConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binary_weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    binary_weight = this->binary_weights_.gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }

      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }

      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (binary_weight == NULL) {
          binary_weight = this->binary_weights_.gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, binary_weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_binary_conv_groups<<<1, 1>>>();
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBinaryConvolutionLayer);

}  // namespace caffe
#endif
