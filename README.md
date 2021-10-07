# Detecting Speaker Personas from Conversational Texts
This repository contains the source code and the dataset for the _EMNLP 2021_ paper [Detecting Speaker Personas from Conversational Texts](https://arxiv.org/pdf/2109.01330.pdf). Jia-Chen Gu, Zhen-Hua Ling, Yu Wu, Quan Liu, Zhigang Chen, Xiaodan Zhu. <br>


## Introduction
Personas are useful for dialogue response prediction. However, the personas used in current studies are pre-defined and hard to obtain before a conversation. To tackle this issue, we study a new task, named Speaker Persona Detection (SPD), which aims to detect speaker personas based on the plain conversational text. In this task, a best-matched persona is searched out from candidates given the conversational text. This is a many-to-many semantic matching task because both contexts and personas in SPD are composed of multiple sentences. The long-term dependency and the dynamic redundancy among these sentences increase the difficulty of this task. We build a dataset for SPD, dubbed as Persona Match on Persona-Chat (PMPC). Furthermore, we evaluate several baseline models and propose utterance-to-profile (U2P) matching networks for this task. The U2P models operate at a fine granularity which treat both contexts and personas as sets of multiple sequences. Then, each sequence pair is scored and an interpretable overall score is obtained for a context-persona pair through aggregation. Evaluation results show that the U2P models outperform their baseline counterparts significantly.

<div align=center><img src="image/task.png" width=50%></div> <br>

<div align=center><img src="image/result.png" width=80%></div>


## Dependencies
Python 3.6 <br>
Tensorflow 1.13.1


## Download
- Download the [BERT released by the Google research](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip), 
  and move to path: ./Pretraining-Based/uncased_L-12_H-768_A-12 <br>
  
- Download the [PMPC dataset](https://drive.google.com/file/d/1sE_N7fi_WojeQBWZcTg4Mw6Pyod27S73/view?usp=sharing) used in our paper,
  and move to path: ```./data_PMPC``` <br>


## Non-Pretraining-Based Models
Train a new model.
```
cd Non-Pretraining-Based/C2P-X/scripts/
bash train.sh
```
The training process is recorded in ```log_train_*.txt``` file. <br>

Test a trained model by modifying the variable ```latest_checkpoint``` in ```test.sh```.
```
cd Non-Pretraining-Based/C2P-X/scripts/
bash test.sh
```
The testing process is recorded in ```log_test_*.txt``` file. A "output_test.txt" file which records scores for each context-persona pair will be saved to the path of ```latest_checkpoint```. Modify the variable ```test_out_filename``` in ```compute_metrics.py``` and then run the following command, various metrics will be shown.
```
python compute_metrics.py
```

You can choose a baseline model by comment/uncomment a model package (from ```model_BOW```, ```model_BiLSTM```, ```model_Transformer``` and ```model_ESIM```) in the first several lines in ```train.py```. The same process and commands can be done for those Non-Pretraining-Based U2P-X Models.


## Pretraining-Based Models
Create the fine-tuning data.
```
cd Pretraining-Based/C2P-BERT/
python data_process_tfrecord.py
```

Running the fine-tuning process.
```
cd Pretraining-Based/C2P-BERT/scripts/
bash train.sh
```

Test a trained model by modifying the variable ```restore_model_dir``` in ```test.sh```.
```
cd Pretraining-Based/C2P-BERT/scripts/
bash test.sh
```

Modify the variable ```test_out_filename``` in ```compute_metrics.py``` and then run the following command, various metrics will be shown.
```
python compute_metrics.py
```

The same process and commands can be done for U2P-BERT.

**NOTE**: Since the dataset is small, each model was trained for 10 times with identical architectures and different random initializations. Thus, we report (mean ± standard deviation) in our paper.


## Cite
If you think our work is helpful, or use the code or dataset, please cite the following paper:
**"Detecting Speaker Personas from Conversational Texts"**
Jia-Chen Gu, Zhen-Hua Ling, Yu Wu, Quan Liu, Zhigang Chen, Xiaodan Zhu. _EMNLP (2021)_
```
@inproceedings{gu-etal-2021-detecting,
 title = "Detecting Speaker Personas from Conversational Texts",
 author = "Gu, Jia-Chen  and
           Ling, Zhen-Hua  and
           Wu, Yu  and
           Liu, Quan  and
           Chen, Zhigang  and
           Zhu, Xiaodan",
 booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
 month = nov,
 year = "2021",
 publisher = "Association for Computational Linguistics",
}
```


## Update
Please keep an eye on this repository if you are interested in our work.
Feel free to contact us (gujc@mail.ustc.edu.cn) or open issues.
