**Railway Track Defect Detection**

Overview

This project focuses on applying Machine Learning and Computer Vision techniques to automatically detect defects in railway tracks from image datasets. Built over a four-week timeline, the project evolved from basic image processing techniques (OpenCV) to a robust Deep Learning pipeline utilizing Transfer Learning and automated hyperparameter tuning.

Technologies & Libraries Used

Python

TensorFlow / Keras (Deep Learning framework)

MobileNetV2 (Pre-trained Convolutional Neural Network)

Optuna (Automated Hyperparameter Optimization)

OpenCV & Matplotlib (Image preprocessing and visualization)

Pandas & NumPy (Data structuring and matrix operations)

Scikit-Learn (Performance evaluation metrics)

Methodology

To combat the challenge of a relatively small training dataset, several advanced data engineering and machine learning techniques were implemented:

Data Organization: Pandas was used to dynamically map file paths and extract class labels (0 for Normal, 1 for Defective) to ensure smooth integration with Keras data generators.

On-the-Fly Image Augmentation: To prevent the model from memorizing the limited dataset (overfitting), ImageDataGenerator was used to apply random rotations, zooms, and horizontal flips directly in memory during training.

Transfer Learning (MobileNetV2): Instead of training a CNN from scratch, the project utilizes Google's MobileNetV2 architecture. By freezing the base layers (which already understand edges, textures, and shapes from 14 million images) and attaching a custom dense classification head, the model learns faster and is highly resistant to background noise (like shadows and gravel).

Hyperparameter Tuning: Optuna was integrated to programmatically search for the optimal learning rate, dense layer unit count, and dropout rate, ensuring maximum validation accuracy without manual guesswork.

Results & Business Logic

The final model achieved a 73% Test Accuracy. However, in safety-critical domains like railway maintenance, overall accuracy is less important than Recall.

The Safety-First Approach:

Defective Recall: 91% The model successfully identified 91% of all actual defective tracks in the blind test set. It learned to prioritize safety above all else.

The Trade-off (False Positives):
To achieve this high safety standard, the model occasionally flags normal tracks with odd shadows as "Defective". In a real-world business context, this is a highly successful outcome: it is much cheaper to send a human inspector to clear a false alarm (False Positive) than it is to miss an actual track defect that could lead to a derailment (False Negative).

How to Run

Train the Model: Run the provided Google Colab pipeline sequentially to preprocess the data, run the Optuna study, and train the model. The best weights will automatically be saved as best_railway_model.keras.

Run Inference: Use the inference.py script to test the model on unseen images:

from inference import predict_track_defect
predict_track_defect("path/to/your/image.jpg")



Timeline Summary

Week 1: Digital image basics, color spaces (BGR to RGB), and initial file processing.

Week 2: Image transformations, feature extraction (Canny Edge Detection, Histograms).

Week 3: Convolutional Neural Networks (CNNs), data augmentation, and generator setup.

Week 4: Transfer Learning integration, Optuna optimization, and final classification reporting.
