#include <vector>

#include "caffe/layers/binary_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BinaryConvolutionParameter binary_conv_param = this->layer_param_.binary_convolution_param();
  bool use_alpha = binary_conv_param.use_alpha();
  bool use_binarization = binary_conv_param.use_binarization();
  // initialization for binary parameters
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const int weight_dim = this->blobs_[0]->count() / this->blobs_[0]->num();
  weight_sum_multiplier_.Reshape(weight_dim,1,1,1);
  caffe_set(weight_sum_multiplier_.count(),Dtype(1),weight_sum_multiplier_.mutable_cpu_data());  
  binary_weights_.ReshapeLike(*this->blobs_[0]);
  alphas_.Reshape(this->num_output_,1,1,1);
  caffe_set(this->num_output_,Dtype(1),alphas_.mutable_cpu_data()); 
  caffe_copy(binary_weights_.count(),weight,binary_weights_.mutable_cpu_data());
  
  // binarize the weights
  if (use_binarization) {
    // compute alpha if needed
    if (use_alpha) {
      caffe_abs(this->num_output_*weight_dim,weight,binary_weights_.mutable_cpu_diff());
      const Dtype* abs_weight = binary_weights_.cpu_diff();
      caffe_cpu_gemv<Dtype>(CblasNoTrans, this->num_output_, weight_dim,
          1. / weight_dim, abs_weight, weight_sum_multiplier_.cpu_data(), 0.,
          alphas_.mutable_cpu_data());
    }
    for (int i = 0; i < this->num_output_; i++) {
      for (int j = 0; j < weight_dim; j++) {
        Dtype binary_code = (weight[i*weight_dim+j]>=0) ? 1:-1;
        binary_weights_.mutable_cpu_data()[i*weight_dim+j] = binary_code*alphas_.cpu_data()[i];
      }
    }
  }
    
  const Dtype* binary_weights = binary_weights_.cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, binary_weights,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}

template <typename Dtype>
void BinaryConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* binary_weights = binary_weights_.cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, binary_weights,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BinaryConvolutionLayer);
#endif

INSTANTIATE_CLASS(BinaryConvolutionLayer);

}  // namespace caffe
