
## Train on IEMOCAP
### Download datasets
* [IEMOCAP download link](https://sail.usc.edu/iemocap/index.html)  


### Extract wav2vec and RoBERTa feature
* See the functions `extract_wav2vec.py` and `extract_roberta.py` .
* Each extracted feature is saved in `.mat` format using `scipy`.  

Modify the argument: `matdir` in `./config/iemocap_feature_config.json` to the folder path of your extracted feature.

### Train model
Set the hyper-parameters on `./config/config.py` and `./config/model_config.json`.
Next, run:
```
python train.py
```


