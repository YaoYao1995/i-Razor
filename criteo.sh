#!/bin/sh
for bs in 500;do
    cp tf_main_retrain_model.py criteo_retrain_bs_$bs.py
    sed -i "s/criteo_bs/$bs/g" criteo_retrain_bs_$bs.py
    sed -i "s/data_name = 'avazu'/data_name = 'criteo'/g" criteo_retrain_bs_$bs.py
    nohup python -u criteo_retrain_bs_$bs.py > ~/i-Razor/logs/criteo_retrain_bs_$bs.log 2>&1 &
    sleep 10
    rm criteo_retrain_bs_$bs.py
done
