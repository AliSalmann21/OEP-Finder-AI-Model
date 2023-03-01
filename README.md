# Machine Learning Pipeline for Binary Classification of Portable Executable (PE) Files

This code implements a machine learning pipeline for binary classification of Portable Executable (PE) files. PE files are a type of executable file format used in Windows operating systems.


## Dataset

The dataset used for this project is stored in a CSV file called `pe_files_dataset.csv`. It contains features extracted from PE files, including file size, number of sections, number of imports, and other attributes relevant to the binary classification task. The dataset also includes a label indicating whether the file is obfuscated (1) or Non-obfuscated (0).


## Machine Learning Pipeline

The machine learning pipeline consists of the following steps:

1. Load the dataset using pandas.
2. Split the dataset into training and testing sets using `train_test_split` function from scikit-learn.
3. Define a function called `create_model` that creates a neural network model using the Keras API.
4. Use grid search with cross-validation to find the optimal hyperparameters for the neural network model using the `GridSearchCV` function from scikit-learn.
5. Train the final neural network model using the best hyperparameters found in step 4.
6. Ensemble learning is used to improve the performance of the model.
7. Feature engineering is performed to improve the performance of the model.


## Detailed Technical Steps:

1. Split the dataset into training and testing sets using the train_test_split function from the scikit-learn library. The testing set size is set to 20% of the total dataset, and a random seed is used for reproducibility.

2. Define a function called create_model that creates a neural network model using the Keras API. The function takes four hyperparameters as input: the number of hidden layers, the number of neurons per layer, the learning rate, and the dropout rate. The model consists of a sequence of dense layers with ReLU activation functions and a final sigmoid activation function for binary classification. The optimizer used is the Adam optimizer, and the loss function is binary cross-entropy. The function returns the compiled Keras model.

3. Use grid search with cross-validation to find the optimal hyperparameters for the neural network model. The hyperparameters to search over are the number of hidden layers, the number of neurons per layer, the learning rate, and the dropout rate. The GridSearchCV function from scikit-learn is used to perform the search. The best hyperparameters are stored in grid_search.best_params_.

4. Train the final neural network model using the best hyperparameters found in step 4. The model is trained for 50 epochs with a batch size of 32, and the testing set is used for validation. The trained model is then used to predict the labels of the testing set, and the confusion matrix and classification report are printed to evaluate the performance of the model.

5. Ensemble learning is used to improve the performance of the model. Five neural network models are trained using the best hyperparameters found in step 4, with each model being trained on a randomly augmented version of the training set using data augmentation techniques such as flipping and rotation. The models are then used to predict the labels of the testing set, and the predictions are combined using majority voting to obtain the final prediction. The confusion matrix and classification report are printed to evaluate the performance of the ensemble model.

6. Feature engineering is performed to improve the performance of the model. Two sets of features are extracted from the PE files: n-gram features and image features. The n-gram features are obtained by extracting all contiguous sequences of bytes of length n from the PE file, hashing them, and counting their frequency. The image features are obtained by treating the byte sequence as a grayscale image and resizing it to 64x64 pixels. The image is then flattened to a 4096-dimensional vector. The n-gram and image features are concatenated with the original feature set, and the resulting feature matrix is used to train the final model.

7. Finally, the final model is trained using the augmented feature matrix, and the performance is evaluated using the confusion matrix and classification report.

## Dependencies

Python 3.6+

Keras 2.3.1+

scikit-learn 0.22.2+

numpy 1.16.2+

pandas 0.24.2+

matplotlib 3.0.3+

seaborn 0.9.0+

## Credits

This project was created by [Ali Salman](https://github.com/AliSalmann21 as part of a Automating the finding of an OEP of the program.

## License

This project is licensed under the MIT License - see the included LICENSE file for details.```
