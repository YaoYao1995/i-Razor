pip install gdown
# avazu 5.2G (after uncompressed: 4.4G Avazu; 790M    part0.tar)
gdown --folder https://drive.google.com/drive/folders/1vVeQeKZtXAurAMIoOOju3sNK40gnUUkO
cd avazu
tar -xvf part0.tar
# criteo  (after uncompressed: 17G Criteo-8d, 5.6G    all.tar)
cd ../
gdown --folder https://drive.google.com/drive/folders/19T7lPhHT4dTs-giulgyCtDbMYxug-s-7
cd criteo-8d
cat part*.tar > all.tar
tar -xvf all.tar