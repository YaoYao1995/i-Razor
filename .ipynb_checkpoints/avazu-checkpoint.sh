#!/bin/sh
for bs in 128;do
    cp tf_main_retrain_model.py avazu_retrain_bs_$bs.py
    sed -i "s/avazu_bs/$bs/g" avazu_retrain_bs_$bs.py
    nohup python -u avazu_retrain_bs_$bs.py > ~/i-Razor/logs/avazu_retrain_bs_$bs.log 2>&1 &
    sleep 10
    rm avazu_retrain_bs_$bs.py
done
