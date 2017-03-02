#include <algorithm>
#include <vector>
#include <cfloat>
#include "caffe/layers/quant_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void QuantLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  forward_func_ = this->layer_param_.quant_param().forward_func();
  backward_func_ = this->layer_param_.quant_param().backward_func();
  if (forward_func_ == "hwgq") {
    int num_centers = this->layer_param_.quant_param().centers_size();
    CHECK_GT(num_centers,0);
    centers_.Reshape(num_centers,1,1,1);
    for (int i = 0; i < num_centers; i++) { 
      centers_.mutable_cpu_data()[i] = this->layer_param_.quant_param().centers(i);
      CHECK_GT(centers_.cpu_data()[i],0);
    }
    // thresholds, the middle point of two continuous positive centers
    thrs_.Reshape(num_centers,1,1,1);    
    for (int i = 0; i < num_centers-1; i++) { 
      thrs_.mutable_cpu_data()[i] = 0.5*(centers_.cpu_data()[i]+centers_.cpu_data()[i+1]);
    }
    thrs_.mutable_cpu_data()[num_centers-1] = FLT_MAX;
  }
}
    
template <typename Dtype>
void QuantLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count(); CHECK_EQ(count,top[0]->count());
  caffe_set(count, Dtype(0), top_data);
  Dtype negative_slope = this->layer_param_.quant_param().negative_slope(); 
  
  if (forward_func_ == "sign") {
    for (int i = 0; i < count; ++i) {
      top_data[i] = (bottom_data[i]>=0) ? Dtype(1):Dtype(-1);
    }
  } else if (forward_func_ == "hard_tanh") {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::min(std::max(bottom_data[i],Dtype(-1)),Dtype(1));
    }
  } else if (forward_func_ == "hwgq") {
    const Dtype* centers_data = centers_.mutable_cpu_data();
    const Dtype* thrs_data = thrs_.cpu_data();
    for (int i = 0; i < count; ++i) {
      if (bottom_data[i]<=0) {
        top_data[i] = -negative_slope;
      } else {
        for (int j = 0; j < thrs_.count(); j++) {
          if (bottom_data[i]<=thrs_data[j]) {
            top_data[i] = centers_data[j]; break;
          }
        }
      }
    }
  } else if (forward_func_ == "relu") {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  } else {
    CHECK(false) << "Unknown Forward Function!";
  }
}

template <typename Dtype>
void QuantLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    float clip_thr = this->layer_param_.quant_param().clip_thr();
    clip_thr = (clip_thr>0) ? clip_thr:FLT_MAX; 
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.quant_param().negative_slope();
    if (backward_func_ == "hard_tanh") {
      for (int i = 0; i < count; ++i) {
        if (-clip_thr<=bottom_data[i] && bottom_data[i]<=clip_thr) {
          bottom_diff[i] = top_diff[i];
        } else {
          bottom_diff[i] = 0;
        }
      }
    } else if (backward_func_ == "relu") {
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
            + negative_slope * (bottom_data[i] <= 0));
        bottom_diff[i] = bottom_diff[i] * (bottom_data[i] <= clip_thr);
      }
    } else {
      CHECK(false) << "Unknown Backward Function!";
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(QuantLayer);
#endif

INSTANTIATE_CLASS(QuantLayer);
REGISTER_LAYER_CLASS(Quant);

}  // namespace caffe
