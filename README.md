## Machine Learning Based Sensing of Particle Shape and Size Using Passive Artificial Cilia

This repsoitory contains the training and testing data and the corresponding python scripts.
There are two directories for two models using the top and bottom cilia perturbations.

In each of these directories the content is similar and is described below.
The following is a description of the files
1. training_models_a.py: Trains saves the ml model using data in the train_data directory to predict semi-major axis
2. training_models_r.py: Trains saves the ml model using data in the train_data directory to predict aspect ratio
3. lstm_models.py: Contains necessary function needed by 1. and 2.
4. predict_values_a.py: Loads and evaluates the model to predict a values using the test set and plots the graph in images directory
5. predict_values_r.py: Loads and evaluates the model to predict r values using the test set and plots the graph in images directory
6. lstm_models_working.py: Contains necessary function needed by 4. and 5.
7. a_lstm: Saved trained model for predicting semi-major axis
8. ar_lstm: Saved trained model for predicting aspect ratio
 
