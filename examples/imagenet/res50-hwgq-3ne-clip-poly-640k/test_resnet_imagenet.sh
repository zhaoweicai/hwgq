
GLOG_logtostderr=1 ../../../build/tools/caffe test \
  --model=train_val.prototxt \
  --weights=resnet50_iter_640000.caffemodel \
  --iterations=2000 \
  --gpu=0  2>&1 | tee log_test_640k.txt