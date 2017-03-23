#!/usr/bin/env sh
set -e

GLOG_logtostderr=1 ./build/tools/caffe train --solver=examples/mnist/hwgq-lenet-solver.prototxt 2>&1 | tee examples/mnist/log-hwgq-lenet.txt $@
