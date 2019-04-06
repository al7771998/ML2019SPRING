#!/bin/bash
curl -OL https://github.com/al7771998/ML2019SPRING/releases/download/0.0.0/model_final_3.h5
python3 cnn_test.py $1 $2