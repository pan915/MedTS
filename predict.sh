#!/bin/bash
python3 -u ./src/predict.py --dataset ./data \
--eval_data './data/dev.pkl' \
--model_dir m0623 \
--model {7}_{10.16915122654289}.model \
--output_dir 'output' \
--cuda \
--cuda_device_num 3 \