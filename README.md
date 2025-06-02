# Spotify Genre Classification - FML Capstone

This project involves building a multi-class classification model using a dataset of ~50,000 Spotify tracks, each labeled with a genre and described by audio features like tempo, energy, and danceability. The goal is to predict music genres accurately using various machine learning models after preprocessing and dimensionality reduction.

## Key Steps

- **Data Preprocessing**:  
  Cleaned missing values using genre-wise medians, one-hot encoded categorical variables, and standardized features.

- **Dimensionality Reduction**:  
  Applied PCA, t-SNE, and LDA to visualize trends and structure in the dataset.

- **Modeling Approaches**:
  - **Random Forest**: AUC = 0.93, Accuracy ≈ 60%
  - **AdaBoost**: AUC ≈ 0.91
  - **Neural Network**: Best model with 2 hidden layers, dropout, and batch normalization reached AUC = 0.9313

## Tools & Libraries

- Python, scikit-learn, PyTorch
- PCA, t-SNE, LDA
- Random Forest, AdaBoost, Feedforward Neural Network

## Outcome

The neural network outperformed tree-based models in AUC, benefiting from advanced architecture and regularization. Feature preprocessing and correct handling of missing/categorical data were critical to performance.

