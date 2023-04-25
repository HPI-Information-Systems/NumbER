# ACM SIGMOD Programming Contest 2021

## Team BoomBoomChicken

Rutgers University - the State University of New Jersey

Member: Chaoji Zuo

Advisor: Prof. Dong Deng

## Description

In the training part, We refine some features (e.g. *brand*, *model*) from the original long text (e.g. *title*, *name*). Apply blocking based on some distinct features (e.g. *size*, *brand*), than train a binary random forest classifier using the label data.

In the execution part, we do the same data preprocessing as training and apply the random forest classifiers we learned before.

## EXECUTION

Just execute **apply_model.py** file. Notice that **utils.p** and **X<sub>i</sub>.csv** also should be put with **apply_model.py** in the same directory.

## Retrain Model

If you want to train a new model, you can execute **train_model_brand_X2X3.py** and **train_model_brand_X4.py** in train_model folder.
