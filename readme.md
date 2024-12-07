# Project Overview

## Use Case Selected
The use case selected for this project is Pose Detection & Correction. I aimed to make a model that can detect the yoga pose and correct the user, by providing feedback on how to improve the pose.

## Approach
My approach to this project was to train a pre-trained model on a dataset of yoga poses. I chose the ResNet18 model, which is a popular choice for image classification tasks. I fine-tuned the model on the yoga pose dataset to improve its performance on this specific task. Using this model, I was able to detect the yoga pose from an image, and also in real-time using a webcam. For each pose, I had pre-defined the angles that the knees and elbows should make, and compared these angles with the angles detected by the model. If the angles were not within a certain threshold, the model would provide feedback on how to correct the pose. There was also a 15 degree tolerance for the angles, to account for slight variations in the pose. The model was trained and able to detect 5 different yoga poses: Warrior, Tree, Downward Dog, Plank, and Goddess.

## Data Preprocessing
The data I used for this project was a dataset of images of people performing yoga poses. I got the data from https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset. The dataset contained images of 5 different yoga poses: Warrior, Tree, Downward Dog, Plank, and Goddess. I preprocessed the data by resizing the images to 224x224 pixels, normalizing the pixel values, and splitting the data into training and testing sets. I also augmented the data by random horizontal flips. This helped to increase the size of the dataset and improve the model's performance.

## Model Architecture
Since I used a pre-trained ResNet18 model, the architecture of the model was already defined. ResNet18 is a convolutional neural network that has 18 layers, including convolutional layers, batch normalization layers, and ReLU activation functions. The model was pre-trained on the ImageNet dataset, which helped it learn features that are useful for image classification tasks. I fine-tuned the model on the yoga pose dataset by replacing the final fully connected layer with a new layer that has 5 output units, one for each yoga pose. I trained the model using the Adam optimizer and the cross-entropy loss function.

## Results
The accuracy of the model on the test data was 95.96%. This was also further tested by using a webcam to detect the yoga poses in real-time. The model was able to detect the yoga poses accurately and provide feedback on how to correct the pose if needed.

## Next Steps
In the future, I would like to improve the model by adding more yoga poses to the dataset and fine-tuning the model on the larger dataset. I would also like to explore other pre-trained models and see if they can improve the performance of the model. Additionally, I would like to deploy the model as a web application so that users can upload images of themselves performing yoga poses and get feedback on how to improve their form. I would also plan on making the UI more user-friendly and interactive.