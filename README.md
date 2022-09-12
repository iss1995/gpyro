# GPyro: uncertainty-aware temperature predictions for additive manufacturing
This repo contains an implementation of the paper _GPyro: uncertainty-aware temperature predictions for additive manufacturing_.
The current implementation is much faster than the one originally used to generate the results in the paper. Moreover, the models' performance is similar.

An example of usage can be found in the main.py file. To change folder paths folders, hyperparmaters etc. edit the function data_processing._config.config().
There is also a hyperparameter optimization algorithm based on optuna.
To install the dependencies you can run:

> pip install -r requirements.txt

The data used in the original paper can be found in fig share ([LINK](https://figshare.com/articles/dataset/GPyro-TD_zip/21063190)). Unzip the data in the project folder.

