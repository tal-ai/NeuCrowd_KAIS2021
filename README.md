# NeuCrowd
This repository is the implementation of *'NeuCrowd: Neural Sampling Network for Representation Learning with Crowdsourced Labels'*

# Dependencies

* Python == 3.6
* tensorflow == 1.8
* scikit-learn (for matrics evaluation)

To develop locally, please follow the instruction below:

    git clone git@github.com:tal-ai/NeuCrowd_KAIS2021.git
    cd NeuCrowd_KAIS2021

# Data Preparation

Please convert your data into a csv data file of correct format,which contains a *label* column which stand for the category(0/1).
The rest columns are the feature columns. Two example data sets are stored in `data` folder.

# Train & Test

You can train the model as follow:

    cd src/
    python trainRLL.py your_data_set_path

You can test the model as follow:

    python inference_neucrowd.py your_data_set_path

# Citation

If you use this code in your research, you can cite our arxiv paper:
    
    @misc{hao2021neucrowd,
          title={NeuCrowd: Neural Sampling Network for Representation Learning with Crowdsourced Labels}, 
          author={Yang Hao and Wenbiao Ding and Zitao Liu},
          year={2021},
          eprint={2003.09660},
          archivePrefix={arXiv},
          primaryClass={cs.LG}
    }

# Contact

If you have any problem to the project, please feel free to report them as issues.

