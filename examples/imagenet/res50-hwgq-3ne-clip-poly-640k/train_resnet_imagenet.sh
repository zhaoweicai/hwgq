
GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt \
  --gpu=0,1,2,3,4,5,6,7  2>&1 | tee log.txt