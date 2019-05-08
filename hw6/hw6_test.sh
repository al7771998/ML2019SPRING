#!/bin/bash
wget -c "https://www.dropbox.com/s/pfunz0xt3kv17s3/rnn12.h5?dl=0" -O rnn12.h5
python3 hw6_test.py $1 $2 $3