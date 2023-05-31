# Instruction to reproduce i-Razor

Code to reproduce the experiments in “*i-Razor: A Differentiable Neural Input Razor for Feature Selection and Dimension Search in DNN-Based Recommender Systems*”.

Our code is mainly implemented with reference to the Tensorflow implementation of AutoFIS. The URL is: https://github.com/zhuchenxv/AutoFIS 

## Dataset
Due to the space constraint, we only provide a small portion of the sampled data from the Avazu dataset as an example. For the acquisition and processing of the complete dataset, we follow the PIN setup.  We refer interested readers to the following URL for more details: https://github.com/Atomu2014/Ads-RecSys-Datasets. For consistency of evaluation, we directly reuse the preprocessed data accessible from the PIN paper. Please refer to  the script [download_criteo_and_avazu.sh](./download_criteo_and_avazu.sh) to obtain the processed data.

## Usage
Our model i-Razor has two stages: the pretraining stage and  the retraining stage. To obtain the input configuration for each baseline, run the script corresponding to each method as follows:

```python
python tf_main_autodim.py  # for AutoDim
python tf_main_irazor.py # for i-Razor
python tf_main_autofis.py # for AutoFIS
python tf_main_darts.py # for DARTS
python tf_main_autofield.py # for AutoField
python tf_main_adafs.py # for AdaFS
...
```

After getting the input configuration, copy it to the “tf_main_retrain_model.py” file, and run it to retrain the model.

```python
# Retrain the model with the optimized input configuration 
python tf_main_retrain_model.py 
```

## Training Device
We experiment with a distributed CPU cluster containing 40 Intel(R) Xeon(R) Platinum 8260 CPUs per working node. The training time and the number of devices used in the pretraining stage of i-Razor are roughly given as follows.

|           | Working Nodes | Training duration |
| --------- | ------------- | ----------------- |
| Avazu     | 35            | 40 min            |
| Avazu-FG  | 35            | 60 min            |
| Criteo    | 40            | 90 min            |
| Criteo-FG | 40            | 270 min           |

