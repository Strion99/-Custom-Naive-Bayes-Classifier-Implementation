# -Custom-Naive-Bayes-Classifier-Implementation
Engineered a probabilistic classifier without using ML libraries, implementing Maximum Likelihood Estimation (MLE) and Laplace Smoothing for robust predictions. Designed the algorithm to process both categorical and continuous variables within the same dataset.


1- Probabilities (Theoretical) 
Let the random variable X follow the distribution:
f(x;θ) = θ2(x +1)(1−θ)x ,x = 0,1,2..,θ ∈ [0,1]
a.  Find the expression describing the MLE estimators for θ for N indepen
dent identically distributed (i.i.d.) samples. How can you be sure that this value
you found is indeed the maximum?
b.  Calculate θ for f(x;θ) using the formula calculated in the first step, and
applying it to the following 15 samples:
[3.2, 1.4, 2.2, 7, 0.5, 3.3,9,0.15,2,3.21,6.13,5.5,1.8,1.2,11]

2- Naive Bayes (Theoretical) 
Consider Table 1, presenting a dataset with 7 samples, each com
prised of three Boolean variables x, y and z, and a Boolean target
variable U. You will use this data to train a Na¨ıve Bayes classifier
and predict U. Specifically:
a.  After learning is complete, what would be the pre
dicted probability P(U = 0|x = 0,y = 0,z = 1)?
b.  Why — in this case — did we not need to exploit the
Laplace trick to solve for question a?
Hint: Try to solve for P(U = 0|x = 0, y = 0, z = 0) and check which
element in the Na¨ıve Bayes classifier formula causes the method to
fail.
c.  Using the probabilities obtained during the Bayes
Classifier training, what would be the predicted probability P(U =
1|x = 0)?
Table 1:
x y z U
1 1 1 0
0 1 1 0
0 0 1 0
1 0 0 1
0 0 1 1
0 1 0 1
1 1 0 1

3- Na¨ıve Bayes Classifier (Programming) 
NOTE: You are NOT allowed to use any existing implementations of the Naive Bayes
Classifier.
You will have to implement the Na¨ıve Bayes Classifier (NBC) by coding the formulas you
saw in class, recitations and the reading material. The classifier should be able to handle
either categorical (i.e. discrete values) or continuous variables. The implementation of
the classifier will be split in 2 functions, one for training and one for predicting.
a.  Implement the NBC training function:
def train NBC(X, X dtype, Y, L, D categorical)
Inputs:
• X: IxM matrix of variables. Rows correspond to the I samples and columns
to the M variables.
• X dtype : String describing the data type of X, which could be either ”cate
gorical” or ”continuous”.
• Y: Ix1 vector. Y is the class variable you want to predict.
• L : Scalar. L is the parameter referred to in the MAP estimates equation.
For L = 0 you get the MLE estimates. L ≥ 0.

• D categorical: 1xM vector. Each element D(m) contains the number of pos
sible different values that the categorical variable m can have. This vector is
ignored if X dtype = ”continuous”.
Output:
• Model: This model should contain all the parameters required by the NBC
to classify new samples. It is up to you to decide its structure. The only
requirement is that it is compatible with your next function.
Notes − Categorical values:
• All categorical variables take values starting from 0. If a variable can take K
possible values, its values are in [0, K-1]. This holds for both the class values
in Y and the values in X.
• If some combinations of values do not occur in the data they take probability
0, unless L is greater than 0.
• D categorical − It is important to pass this information to the function be
cause the samples used for training do not necessarily contain all possible
values that the variables can take.
b.  Implement the NBC prediction function:
def predict NBC(model, X, X dtype)
Inputs:
• model: The model previously trained using train NBC.
• X: JxM matrix of variables. Rows correspond to the J samples and columns
to the M variables.
• X dtype : String describing the data type of X, which could be either ”cate
gorical” or ”continuous”.
Output:
• predictions: Jx1 vector. Contains the predicted class for each input samples.
c.  Assess the classifier using the datasets uploaded along with the as
signment.
The analysis shall consist of the following steps:
− Randomly split each dataset into two parts, one containing 75% of the
samples (training set) and one containing the remaining 25% (test set).
− Train the classifier on the training set. Train the algorithm for the categori
cal/continuous data on the corresponding datasets.

− Perform predictions on the test datasets and assess the model’s accuracy, i.e.
the percentage of correctly classified samples.
− Repeat the procedure 100 times, each time randomly re-splitting the dataset
into 75% train and 25% test sets. Compute the average accuracy of the
algorithm.
Note: The content of the attached csv files is the following:
a) Dataset* X categorical: categorical variable data
b) Dataset* X continuous: continuous variable data
c) Dataset* Y: class labels for the corresponding dataset
d) Dataset* D categorical: number of possible values that a feature might have
d. Bonus question: [10 points] How does the choice of the hyperparameter L affect
the results, in the case of the categorical classification? Experiment with small
and large values of L
