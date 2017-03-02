#include <algorithm>
#include <vector>
#include <cfloat>
#include "caffe/layers/quant_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void QuantSignForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] >= 0 ? Dtype(1):Dtype(-1);
  }
}

template <typename Dtype>
__global__ void QuantHardTanhForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = min(max(in[index],Dtype(-1)),Dtype(1));
  }
}

template <typename Dtype>
__global__ void QuantHwgqForward(const int n, const Dtype* in, Dtype* out,
    const Dtype* thrs, const Dtype* centers, int num_thrs, Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    if (in[index] <= 0) {
       out[index] = -negative_slope;
    } else {
      for (int j = 0; j < num_thrs; j++) {
        if (in[index] <= thrs[j]) {
          out[index] = centers[j]; break;
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void QuantReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void QuantLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count(); CHECK_EQ(count,top[0]->count());
  caffe_gpu_set(count, Dtype(0), top_data);
  Dtype negative_slope = this->layer_param_.quant_param().negative_slope();
  
  if (forward_func_ == "sign") {
    QuantSignForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data);
  } else if (forward_func_ == "hard_tanh") {
    QuantHardTanhForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data);
  } else if (forward_func_ == "hwgq") {
    int num_thrs = thrs_.count();
    const Dtype* thrs_data = thrs_.gpu_data();
    const Dtype* centers_data = centers_.gpu_data();
    QuantHwgqForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, thrs_data, centers_data, num_thrs, negative_slope);
  } else if (forward_func_ == "relu") {
    QuantReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, negative_slope);
  } else {
    CHECK(false) << "Unknown Forward Function!";
  }
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void QuantHardTanhBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, float thr, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (-thr<=in_data[index] && in_data[index]<=thr) {
      out_diff[index] = in_diff[index];
    } else {
      out_diff[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void QuantReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype clip_thr) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * negative_slope);
    out_diff[index] = out_diff[index] * (in_data[index] <= clip_thr);
  }
}

template <typename Dtype>
void QuantLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    float clip_thr = this->layer_param_.quant_param().clip_thr();
    clip_thr = (clip_thr>0) ? clip_thr:FLT_MAX; 
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.quant_param().negative_slope();
    if (backward_func_ == "hard_tanh") {
      QuantHardTanhBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_data, clip_thr, bottom_diff);
    } else if (backward_func_ == "relu") {
      QuantReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, bottom_data, bottom_diff, negative_slope, clip_thr);
    } else {
      CHECK(false) << "Unknown Backward Function!";
    }
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(QuantLayer);


}  // namespace caffe
