# NeuCrowd
This repository is the implementation of *'NeuCrowd: Neural Sampling Network for Representation Learning with Crowdsourced Labels'*.

# Dependencies

* Python == 3.6
* tensorflow == 1.8
* scikit-learn (for matrics evaluation)

To develop locally, please follow the instruction below:

    git clone git@github.com:tal-ai/NeuCrowd_KAIS2021.git
    cd NeuCrowd_KAIS2021

# Data Preparation

Please convert your data into three csv data files (train.csv, valid.csv, test.csv) of correct format, each of which contains a *label* column which stand for the category(0/1).
The rest columns are the feature columns. 

Two example data sets used in the paper are stored in `data` folder.

# Train & Test

You can train the model as follow:

    cd src/
    python trainRLL.py your_data_set_path

You can test the model as follow:
    
    cd src/
    python inference_neucrowd.py your_data_set_path

# Contact

If you have any problem to the project, please feel free to report them as issues.

