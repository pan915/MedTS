#!/bin/bash
python -u ./src/train.py --dataset ./data \
--train_data ./data/train.pkl \
--epoch 100 \
--save m0623 \
--cuda \
--cuda_device_num 0 \