#!/bin/bash
wget -c "https://www.dropbox.com/s/ki5hs6qsjhvkvc3/model_final_3.h5?dl=1" -O model_final_3.h5
python3 hw4.py $1 $2