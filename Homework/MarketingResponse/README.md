![](UTA-DataScience-Logo.png)

# Project Title

* This repository applies machine learning techniques and models to springlife marketing response to predict customer responses to the direct mail (DM) that springleaf sends out to connect with their current and potential clientale.
From Kaggle's "Springleaf Marketing Response" [(https://www.kaggle.com/competitions/springleaf-marketing-response/overview)]. 

## Overview

* Springleaf is a financial services company that provides customers with personal and auto loans. Direct mail is Springleaf's primary communication source to connect with their current customers and reach out to potential and target customers.
* The task, as defined by the Kaggle challenge is to develop a model to "determine whether to send a direct mail peice to a customer". This repository approaches this problem as a binary classification task, using models x and were compared against each other. x was the best model for the task as it was able to determine whether a customer succesfully responded to a DM and hence should be futher contacted via DM scored at ~x% accuracy. At the time of this writing, the best performance on the Kaggle leaderboards of this metric is x%.

## Summary of Work Done

### Data

* Data:
  * Type: Binary Classification
    * Input: CSV file: train.csv, test.csv; described customer response
    * Output: sucess or failure based on whether or not the customer responded or not -> target col = 'x'
  * Size: Original training and testing datasets together was x MB (training: x rows & x features (x MB); test: x rows & x features (x MB). After cleaning and proper preprocessing both datasets together was about x MB.
  * Instances (Train, Test, Validation Split): training: x, testing: x.
[work in progress]

#### Preprocessing / Clean up

[work in progress]

#### Data Visualization

[work in progress]

### Problem Formulation

* Train information about demographics, diagnosis and treatment options, insurance and more with machine learning to provide a better view about aspects that may contribute to health equity.
  * Models
    * RandomForest; chosen for it's ease and flexibility and hence used as a base model for comparison.
    * Catboost; chosen for it's built-in methods, predictive power and great results without the need for parameter tuning, and robustness.
  * No in-depth fine-tuning or optimization to the models such as hypyerparameters, feature importance or cross validation were done. 
[work in progress]

### Training

* Describe the training:
  * Training was done on a Surface Pro 9 using Python via jupyter notebook.
  * Training did not take long to process, with the longest training time to be approximately a minute.
  * Concluded training when results were satisfactory and plenty of evaluation metrics for comparison observed fairly decent results.
[work in progress]

### Performance Comparison

* Key performance metrics were imported from sklearn and consist of:
  * log_loss().
  * classification_report().
  * accuracy_score().
  * roc_auc_score().
  * roc_curve().
  * auc().
[work in progress]

### Conclusions

[work in progress] 

### Future Work

[work in progress]

## How to reproduce results

* The notebooks are well organized and include further explanation; a summary is provided below:
* Download the original data files ('train.csv', 'test.csv') from Kaggle or directly through the current repository along with the processed data files.
* Install the necessary libraries
* Run the notebooks attached
* As long as a platform that can provide Python, such as Collab, Anaconda, etc, is used, results can be replicated.

### Overview of files in repository

* The repository includes x files in total.
  * training.csv: Official and original training dataset that was provided from Kaggle
  * test.csv: Official and original test dataset that was provided from Kaggle
[work in progress]

### Software Setup
* Required Packages:
  * Numpy
  * Pandas
  * Sklearn
  * Seaborn
  * Matplotlib.pyplot
  * Math
  * Catboost
  * Scipy
  * Tabulate
* Installlation Proccess:
  * Installed through Linux subsystem for Windows
  * Installed via Ubuntu
  * pip3 install numpy
  * pip3 install pandas
  * pip3 install -U scikit-learn
  * pip! install catboost

### Data

* Data can be downloaded through the official Kaggle website through the link stated above. Or through Kaggle's API interface. Can also be downloaded directly through the datasets provided in this directory.

### Training

* Models can be trained by first splitting the testing dataset into two datasets to be trained and validated. Choose the model you wish to train and fit the data and validation variables. Look below in citations to research official websites to find parameters of the model functions to tune to your liking.

#### Performance Evaluation

* Evaluation metrics are imported such as the log loss, accuracy score, classification score. The ROC curve and AUC measurement were also imported and then placed into a function for comparison of multiple models.
* Run the notebooks.


## Citations
 [work in progress]