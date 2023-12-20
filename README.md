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
* Python: 3.9.13
* tensorflow: 2.9.1
* tflite_runtime: 2.14.0
* keras_image_helper: 0.0.1


Project dependencies can be installed by running:
```python
pip install -r requirements.txt
```


This experiment was run on saturn cloud [saturn.io](https://saturncloud.io/), using single TPU.

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

## Model Deployment On Local
The bean leaf model is served through aws lambda function. For that purposes, the first step is prepare the script and test it on local. Below is step by step how to do it:
1. Minimize trained model  ([xception_bean_leaf_16_0.947.h5](./saved-model/xception_bean_leaf_16_0.947.h5)) through convert it to more lightweight version in tflite model. Full implementation of the conversion can refer this [notebook](./conversion_to_tflite.ipynb). The process will result an .tflite model, that can be found in [here](./bean-leaf-model.tflite)
2. After it we need to prepare script for get the image input , throw it to the model and return the prediction score as a response. The script is written on [lambda_function.py](./lambda_function.py). The script basically contain predict function that will preprocess input image, interprete the tflite model, put the input to the model and return the model prediction scores.
3. Test the lambda_function with simply import the script function like below image. if its implemented correctly , it will return model prediction. For the input, its accept image url. you can host your image first on any file hosting provider or simpy use this sample image: https://drive.google.com/uc?export=view&id=1MGvOaIy94muwFCofOd88pNRszUUiwdvf. For generate  image direct link in gdrive, check this [url](https://www.makeuseof.com/create-direct-link-google-drive-files/).
   <img width="1003" alt="image" src="https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/256b7f32-4be2-4f15-9351-d0eec198df57">

## Containerize Model Using Docker
For the sake of portability and easeness of model deployment, we need to wrap our model using isoloted environment container (we using Docker here). The step by step:
1. Make sure your computer already have Docker Desktop or install [here](https://www.docker.com/products/docker-desktop/).
2. Build docker image from prepared [Dockerfile script](./Dockerfile)
```python
docker build -t bean-leaf-model .
```
3. Make sure the docker image already created successfully with:
```python
docker images
```
4. Run docker images on port 8080
```python
docker run -it --rm -p 8080:8080 bean-leaf-model:latest
```
5. Test the containerized model through utilizing [test.py](./test.py).
![image](https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/91f7f2c2-5eb7-40c7-b6b9-d30fcf61237a)

## Deploy model on cloud (AWS)
After successfully encapsulating our model within a Docker container, we are now prepared to deploy it to a cloud environment. In this scenario, we leverage cloud computing services offered by AWS, specifically AWS Lambda and AWS API Gateway. AWS Lambda serves as our serverless environment for hosting the model, while AWS API Gateway functions as our REST API service. When a user makes a request to the REST API, API Gateway seamlessly forwards the user's request to our Lambda service. Step by step:
1. Create repository in AWS ECR
```python
aws ecr create-repository --repository-name bean-leaf-tflite-images
```

2. Login to our ECR repo (our container repository service provider in AWS)
```python
aws ecr get-login-password --region <your-aws-region> | docker login --username AWS --password-stdin <your-aws-id>.dkr.ecr.<your-aws-region>.amazonaws.com/bean-leaf-tflite-images
```

3. Push our created docker container on local to the our ECR repo
```python
docker tag <image_name_in_your_local_desktop>:latest <repo_URI>/<repo_name_in_ECR>:latest
```

4. Create Lambda Function with container image as our function option. Select our created ECR repo.
<img width="1312" alt="image" src="https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/214d92ff-7994-4ff5-b410-153229659bb2">

5. Create Rest API with API Gateway. Click "Create API" button to create api and add post resource with "/predict" endpoint route. Last but not least, click "Deploy API" button
<img width="1068" alt="image" src="https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/5538aa90-febe-413e-a4df-60e16c78ba28">
<img width="1068" alt="image" src="https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/c7044440-8243-4860-bf50-4e912566c61e">


6. Test with [test.py](./test.py) script and change the url to your created rest api endpoint.

This project already deployed to AWS that can be accessed on:
```python
 [POST] https://9f12y6gsoe.execute-api.ap-southeast-1.amazonaws.com/production/predict
```
Above link can be tested also using [Postman](https://www.postman.com/) tool. 
![image](https://github.com/ryanpram/bean-leaf-lesions-classification/assets/34083758/ba2cc273-25e2-4f80-9dc8-cea5a47f3e2f)








