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
2. Classification: the algorithm was applied to the classification problem

## Usage for the user

- Regression folder:

  1. Run the script _main.py_ from the command line to obtein maml and pretrained model results:
   ```sh
        python main.py -k <number of datapoints (choise 5 or 10)> 

    ```
- Classification folder:  
