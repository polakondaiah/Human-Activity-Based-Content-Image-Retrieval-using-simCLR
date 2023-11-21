# Human-Activity-Based-Content-Image-Retrieval-using-simCLR



Introduction
This project focuses on developing a Content-Based Image Retrieval (CBIR) system specialized in recognizing and retrieving images based on human activities. Leveraging advanced neural network architectures, the system aims to efficiently rank images in terms of similarity to given queries.

Dataset
Our dataset comprises 11,443 images across 15 human activity classes. The data is divided into training, validation, query, and gallery sets, with each image meticulously mapped to its corresponding activity label.

Dataset Distribution
Training Data: 9,163 images
Validation Data: 2,280 images
Query Data: 150 images
Gallery Data: 1,000 images
Activities covered include calling, clapping, cycling, dancing, and more.


### Accessing the Dataset
Due to the large size of the dataset, it is hosted externally. To access and download the complete dataset, please visit the following link: [Download Dataset](https://drive.google.com/file/d/1de1NZ7OYE71jtipqpd8T96Z1ggji3GVi/view?usp=sharing)

Model Architecture
The core of our project is the SimCLRNetwork, adapted from ResNet50, optimized for contrastive learning. Alongside, we employ the NT-Xent Loss function to enhance our model's capability to generate distinct and informative embeddings for each activity class.

Implementation
Our implementation includes steps for data preparation, model training, and evaluation.

Data Preparation
Functions for loading image information and organizing the dataset are integral to our preparation process.

Model Training
The model undergoes 10 epochs of training with validation steps to ensure accuracy and generalization. Checkpointing and model saving mechanisms are also incorporated.

Results and Evaluation
Our model demonstrates promising results in image retrieval tasks:

Mean Average Precision: 0.21
Precision at 1: 0.38
Precision at 10: 0.83
Precision at 50: 1.0
Mean Rank: 6.9
Visualizations
We include visualizations depicting the system's retrieval accuracy for various human activities, showcasing the model's effectiveness and areas for refinement.

Conclusion
The project successfully establishes a robust CBIR system, highlighting the potential of combining ResNet and SimCLR for image retrieval tasks, particularly in recognizing human activities.

