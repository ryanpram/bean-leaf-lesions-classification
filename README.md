# Bean Leaf Lesions Classification
## Problem and Project Description
Bean crops are susceptible to various diseases, and early detection of these diseases is crucial for ensuring a healthy harvest. One common issue is the development of leaf lesions, which can be caused by diseases such as angular leaf spot, bean rust, or may indicate a healthy leaf. Traditional methods of disease diagnosis can be time-consuming and may not provide timely information for effective intervention.

In this project, we aim to develop a deep learning model for the automated classification of bean leaf lesions into three classes: angular leaf spot, bean rust, and healthy. By leveraging the power of convolutional neural networks (CNNs), we intend to create a robust solution that can accurately identify and classify lesions based on images of bean leaves.

<img width="908" alt="image" src="https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/9f424e67-f813-4037-a467-f5b2b199f809">



## Dataset
The dataset of this project taken from [Kaggle Dataset link](https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification/data). The used dataset also can download from following gdrive link : https://drive.google.com/file/d/1zyce3Y661pJ82PfCe0Kp93N2-Nkz1Var/view?usp=sharing.
Alternatively can run below commmad:
```python
!gdown --id '1zyce3Y661pJ82PfCe0Kp93N2-Nkz1Var' 
```

## Dependencies
To run this project, you will need the following dependencies:
* Python 3.9.13
* tensorflow==2.9.1

## Model Creation
In this project, we employ the Convolutional Neural Network (CNN) architecture, specifically Xception. Xception is a deep convolutional neural network architecture that utilizes Depthwise Separable Convolutions ([reference](https://maelfabien.github.io/deeplearning/xception/)). Additionally, we apply transfer learning techniques using pre-trained weights from 'imagenet'.

## Evaluation Metric
Since the used dataset in this project are balance, so we use accuracy as our evaluation metric. The formula is showed below:

$$\ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} \$$

Where:
- $TP$ (True Positives): The number of samples correctly predicted as positive (correctly classified instances of the positive class).
- $TN$ (True Negatives): The number of samples correctly predicted as negative (correctly classified instances of the negative class).
- $FP$ (False Positives): The number of samples incorrectly predicted as positive (instances of the negative class misclassified as the positive class).
- $FN$ (False Negatives): The number of samples incorrectly predicted as negative (instances of the positive class misclassified as the negative class).

## Best Model
From the exploration, we determined that **Xception** model with dropout regularization modification is the best model. Best model achieving Accuracy scores of **94.7%** for the validation data. Founded best param listed below:
- learning rate: 0.01
- inner size : 100
- droprate : 0.8

For detailed exploration, please refer to [the notebook](./exploration_notebook.ipynb)


