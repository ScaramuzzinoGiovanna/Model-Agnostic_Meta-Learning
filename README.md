# Model-Agnostic_Meta-Learning

## Intro

This repo refers to _Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks_ paper you can find [here](https://arxiv.org/pdf/1703.03400.pdf)

## Installation

1. Clone this repository with:
```sh
git clone https://github.com/ScaramuzzinoGiovanna/Model-Agnostic_Meta-Learning.git
```
2. Prerequisites for run this code:

- __Python 3.8.5__
- __tensorflow 2.4.1__ , __keras 2.4.3__
- __tensorboard 2.4.0__
- __OpenCV__, __Matplotlib__, __Numpy__

It is advised to install tensorflow in an environment (venv or conda) for better management.

## Folders
The MAML algorithm has been implemented

1. Regression: the algorithm was applied to the regression problem
    - models: contains the optimal models
    - plots: contains the plots
    - maml.py: implementation of maml algorithm
    - pretrained.py: model pretraining on all of the tasks
    - main.py: main code to run. Contains the performance evaluate by fine tuning the model of both the MAML model and pretrained model
    - regressionModel.py: regression model
    - sinusoidGenerator.py: generator of tasks (sinusoid functions)

2. Classification: the algorithm was applied to the classification problem
    - dataset: contains Miniimagenet and Omniglot datasets
    - models_miniimagenet and models_omniglot: contain the optimal models
    - maml.py: MAML algorithm
    - maml_firstOrderApprox.py: MAML algorithm with first order approximation
    - model.py: contains models to train
    - taskGenerator.py: generate tasks and batches of tasks from datasets

## Usage for the user

- Regression folder:

  1. Run the script _main.py_ from the command line to obtein maml and pretrained model results:
   ```sh
        python main.py -k <number of datapoints (choise 5 or 10)> 

    ```
- Classification folder:  
   1. Run the script _maml.py_ from the command line:
     ```sh
        python maml.py +  (choose one of the four cases below and choose whether to perform train or test)

             1. 5way 5shot omniglot
             -d omniglot -t train -n_way 5 -shot_num 5 -query_num 5 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel
             -d omniglot -t test -n_way 5 -shot_num 5 -query_num 5 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel

             2. 5way 1shot omniglot
             -d omniglot -t train -n_way 5 -shot_num 1 -query_num 1 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel
             -d omniglot -t test -n_way 5 -shot_num 1 -query_num 1 -update_steps_train 1 -update_steps_test 3 -lr 0.4 -meta_batch_size 32 -model_name OmniglotConvModel

             3. 5way 1shot miniimagenet
             -d miniimagenet -t train -n_way 5 -shot_num 1 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 4 -model_name MiniimagenetConvModel
             -d miniimagenet -t test -n_way 5 -shot_num 1 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 4 -model_name MiniimagenetConvModel

             4. 5way 5shot miniimagenet
             -d miniimagenet -t train -n_way 5 -shot_num 5 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 2 -model_name MiniimagenetConvModel
             -d miniimagenet -t test -n_way 5 -shot_num 5 -query_num 15 -update_steps_train 5 -update_steps_test 10 -lr 0.01 -meta_batch_size 2 -model_name MiniimagenetConvModel
  ```

   2. Run the scripts _maml_firstOrderApprox.py_ from the command line:
     ```sh
        python maml_firstOrderApprox.py +  (choose one of the four cases above and choose whether to perform train or test)

            ```