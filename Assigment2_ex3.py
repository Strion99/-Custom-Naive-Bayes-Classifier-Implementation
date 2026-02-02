import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_NBC(X, X_dtype, Y, L, D_categorical=None):
    # Determine the unique classes in Y
    classes = np.unique(Y)
    num_classes = len(classes)
    num_samples, num_features = X.shape

    # Initialize model dictionary
    model = {'priors': {}, 'conditionals': {}, 'dtype': X_dtype}
    
    # P(Y=y_i|X1...XN)= (P(y=y_i)*P(X1...XN|Y=y_i))/(sum(P(Y=y_j)*P(X1...XN|Y=y_j)))
    if X_dtype == 'categorical':
        # If D_categorical is not passed, raise an error
        if D_categorical is None:
            raise ValueError("D_categorical must be provided for categorical data.")
        
        # Calculate priors P(Y=y_i)
        for c in classes:
            model['priors'][c] = (np.sum(Y == c) + L) / (num_samples + L * num_classes) #π_k
        
        # Calculate conditional probabilities P(X_i = x_ij | Y = y_i) for categorical data
        model['conditionals'] = {c: {} for c in classes}
        for c in classes:
            class_mask = (Y == c)
            class_count = np.sum(class_mask)
            
            for m in range(num_features):
                feature_vals = np.arange(D_categorical[m])  # Possible values of the feature
                model['conditionals'][c][m] = {}
                
                for val in feature_vals:
                    count = np.sum((X[:, m] == val) & class_mask)
                    model['conditionals'][c][m][val] = (count + L) / (class_count + L * D_categorical[m])
               
    elif X_dtype == 'continuous':
        # Calculate priors P(Y=c)
        for c in classes:
            model['priors'][c] = (np.sum(Y == c) + L) / (num_samples + L * num_classes)
        
        # Calculate mean and variance for each feature for continuous data
        model['conditionals'] = {c: {'mean': [], 'var': []} for c in classes}
        
        for c in classes:
            class_mask = (Y == c)
            class_samples = X[class_mask]
            class_count = class_samples.shape[0]
            
            for m in range(num_features):
                # Calculate mean and variance
                mean = np.mean(class_samples[:, m])
                var = np.var(class_samples[:, m])  
                
                model['conditionals'][c]['mean'].append(mean)
                model['conditionals'][c]['var'].append(var)
    
    return model


def predict_NBC(model, X, X_dtype):
    num_samples, num_features = X.shape  
    classes = model['priors'].keys()     
    predictions = []  # To store the predicted classes
    
    # Loop over each sample in X 
    for i in range(num_samples):  
        sample = X[i, :]
        posteriors = {}  

        # Loop over each class
        for c in classes:  
            log_posterior = np.log(model['priors'][c])
            
            if X_dtype == 'categorical':
                # For categorical data, multiply the prior by the conditional probabilities
                for m in range(num_features):
                    feature_value = sample[m]
                    # Get the conditional probability of the feature value given class c
                    log_posterior += np.log(model['conditionals'][c][m][feature_value])
            
            elif X_dtype == 'continuous':
                # For continuous data, use the Gaussian distribution formula for likelihood
                for m in range(num_features):
                    mean = model['conditionals'][c]['mean'][m]
                    var = model['conditionals'][c]['var'][m]
                    # Gaussian probability density function (log form)
                    log_likelihood = -0.5 * np.log(2 * np.pi * var) - ((sample[m] - mean) ** 2) / (2 * var)
                    log_posterior += log_likelihood
            
            posteriors[c] = log_posterior  # Store the computed posterior for this class
        
        # Predict the class with the highest posterior probability
        predicted_class = max(posteriors, key=posteriors.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)  # Return the predictions as a numpy array

# Function to calculate accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Function to perform the evaluation process
def evaluate_classifier(X, Y, X_dtype, D_categorical=None, num_repeats=100):
    accuracies = []
    
    
    for _ in range(num_repeats):
        # Split the dataset: 75% training, 25% testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)
        
        # Train the model using the training set
        model = train_NBC(X_train, X_dtype, Y_train, L=1, D_categorical=D_categorical)
    
        # Make predictions on the test set
        predictions = predict_NBC(model, X_test, X_dtype)

        # Calculate accuracy . Compare your prediction with the Y's in the 25% test data
        acc = accuracy(Y_test, predictions)
        accuracies.append(acc)
        
    
    # Calculate the average accuracy over all splits
    avg_accuracy = np.mean(accuracies)
    

    
    return avg_accuracy 

# Load the datasets
# Adjust the paths to where the CSV files are located
X_categorical = pd.read_csv('DatasetA_X_categorical.csv').values
Y_categorical = pd.read_csv('DatasetA_Y.csv').values.flatten()  # Ensure Y is a 1D array
D_categorical = pd.read_csv('DatasetA_D_categorical.csv', header=None ).values.flatten() # Flatten for easier use

X_continuous = pd.read_csv('DatasetB_X_continuous.csv').values
Y_continuous = pd.read_csv('DatasetB_Y.csv').values.flatten()

# Evaluate the classifier on the categorical dataset
print("Evaluating on categorical dataset...")
avg_accuracy_categorical  = evaluate_classifier(X_categorical, Y_categorical, X_dtype='categorical', D_categorical=D_categorical)
print(f"Average accuracy on categorical data: {avg_accuracy_categorical * 100:.2f}%")


# Evaluate the classifier on the continuous dataset
print("Evaluating on continuous dataset...")
avg_accuracy_continuous  = evaluate_classifier(X_continuous, Y_continuous, X_dtype='continuous')
print(f"Average accuracy on continuous data: {avg_accuracy_continuous * 100:.2f}%")
