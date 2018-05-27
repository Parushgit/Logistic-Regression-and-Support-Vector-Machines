# Logistic-Regression-and-Support-Vector-Machines

Note
A zipped le containing skeleton Python script les and data is provided. Note that for each problem, you
need to write code in the specied function within the Python script le. For logistic regression, do not
use any Python libraries/toolboxes, built-in functions, or external tools/libraries that directly
perform the learning or prediction.. Using any external code will result in 0 points for that problem.
Evaluation
We will evaluate your code by executing script.py le, which will internally call the problem specic
functions. You must submit an assignment report (pdf le) summarizing your ndings. In the problem
statements below, the portions under REPORT heading need to be discussed in the assignment report.
Data Sets
In this assignment, we still use MNIST. In the script le provided to you, we have implemented a function,
called preprocess(), with preprocessing steps. This will apply feature selection, feature normalization, and
divide the dataset into 3 parts: training set, validation set, and testing set.
Submission
You are required to submit a single le called proj3.zip using UBLearns.File proj3.zip must contain 2 les:
report.pdf and script.py
• Submit your report in a pdf format. Please indicate the team members on the top of the report.
• The code le should contain all implemented functions. Please do not change the name of the le.
Using UBLearns Submission: In the groups page of the UBLearns website you will see groups
called CSE574 Project Group x". Please choose any available group number for your group and join
the group. All project group members must join the same group. Please do not join any other group on
UBLearns that you are not part of. You should submit one solution per group through the groups page.
1 Your tasks
• Implement Logistic Regression and give the prediction results.
1
• Use the Support Vector Machine (SVM) toolbox sklearn.svm.SVM to perform classication.
• Write a report to explain the experimental results with these 2 methods.
• Extra credit: Implement the gradient descent minimization of multi-class Logistic Regression (using
softmax function).
1.1 Problem 1: Implementation of Logistic Regression (40 code + 15 report =
55 points)
You are asked to implement Logistic Regression to classify hand-written digit images into correct corre-
sponding labels. The data is the same that was used for the second programming assignment. Since the
labels associated with each digit can take one out of 10 possible values (multiple classes), we cannot directly
use a binary logistic regression classier. Instead, we employ the one-vs-all strategy. In particular, you have
to build 10 binary-classiers (one for each class) to distinguish a given class from all other classes.
1.1.1 Implement blrObjFunction() function (20 points)
In order to implement Logistic Regression, you have to complete function blrObjFunction() provided in the
base code (script.py). The input of blrObjFunction.m includes 3 parameters:
• X is a data matrix where each row contains a feature vector in original coordinate (not including the
bias 1 at the beginning of vector). In other words, X 2 RND. So you have to add the bias into
each feature vector inside this function. In order to guarantee the consistency in the code and utilize
automatic grading, please add the bias at the beginning of feature vector instead of the end.
• wk is a column vector representing the parameters of Logistic Regression. Size of wk is (D + 1)  1.
• yk is a column vector representing the labels of corresponding feature vectors in data matrix X. Each
entry in this vector is either 1 or 0 to represent whether the feature vector belongs to a class Ck or not
(k = 0; 1;    ;K 􀀀 1). Size of yk is N  1 where N is the number of rows of X. The creation of yk is
already done in the base code.
Function blrObjFunction() has 2 outputs:
• error is a scalar value which is the result of computing equation (2)
• error grad is a column vector of size (D + 1)  1 which represents the gradient of error function
obtained by using equation (3).
1.1.2 Implement blrPredict() function (20 points)
For prediction using Logistic Regression, given 10 weight vectors of 10 classes, we need to classify a feature
vector into a certain class. In order to do so, given a feature vector x, we need to compute the posterior
probability P(y = Ckjx) and the decision rule is to assign x to class Ck that maximizes P(y = Ckjx). In
particular, you have to complete the function blrPredict() which returns the predicted label for each feature
vector. Concretely, the input of blrPredict() includes 2 parameters:
• Similar to function blrObjFunction(), X is also a data matrix where each row contains a feature vector
in original coordinate (not including the bias 1 at the beginning of vector). In other words, X has size
N  D. In order to guarantee the consistency in the code and utilize automatic grading, please add
the bias at the beginning of feature vector instead of the end.
• W is a matrix where each column is a weight vector (wk) of classier for digit k. Concretely, W has
size (D + 1)  K where K = 10 is the number of classiers.
The output of function blrPredict() is a column vector label which has size N  1.
2
1.1.3 Report (15 points)
In your report, you should train the logistic regressor using the given data X (Preprocessed feature vectors
of MNIST data) with labels y. Record the total error with respect to each category in both training data
and test data. And discuss the results in your report and explain why there is a dierence between training
error and test error.
1.2 For Extra Credit: Multi-class Logistic Regression (10 code + 10 report =
20 points)
In this part, you are asked to implement multi-class Logistic Regression. Traditionally, Logistic Regression
is used for binary classication. However, Logistic Regression can also be extended to solve the multi-class
classication. With this method, we don't need to build 10 classiers like before. Instead, we now only need
to build 1 classier that can classify 10 classes at the same time.
1.2.1 Implement mlrObjFunction() function (10 points)
In order to implement Multi-class Logistic Regression, you have to complete function mlrObjFunction()
provided in the base code (script.py). The input of mlrObjFunction.m includes the same denition of
parameter as above. Function mlrObjFunction() has 2 outputs that has the same denition as above. You
should use multi-class logistic function to regress the probability of each class.
1.2.2 Report (10 points)
In your report, you should train the logistic regressor using the given data X(Preprocessed feature vectors of
MNIST data) with labels y. Record the total error with respect to each category in both training data and
test data. And discuss the results in your report and explain why there is a dierence between training error
and test error. Compare the performance dierence between multi-class strategy with one-vs-all strategy.
1.3 Support Vector Machines (20 code + 25 report = 45 points)
In this part of assignment you are asked to use the Support Vector Machine tool in sklearn.svm.SVM to
perform classication on our data set. The details about the tool are provided here: http://scikit-learn.
org/stable/modules/generated/sklearn.svm.SVC.html.
1.3.1 Implement script.py function (10 points)
Your task is to ll the code in Support Vector Machine section of script.py to learn the SVM model and
compute accuracy of prediction with respect to training data, validation data and testing using the following
parameters:
• Using linear kernel (all other parameters are kept default).
• Using radial basis function with value of gamma setting to 1 (all other parameters are kept default).
• Using radial basis function with value of gamma setting to default (all other parameters are kept
default).
• Using radial basis function with value of gamma setting to default and varying value of C (1; 10; 20; 30;    ; 100)
and plot the graph of accuracy with respect to values of C in the report.
1.3.2 Report (25 points)
In your report, you should train the SVM using the given data X(Preprocessed feature vectors of MNIST
data) with labels y. And discuss the performance dierences between linear kernel and radial basis, dierent
gamma setting.
3
Appendices
A Logistic Regression
Consider x 2 RD as an input vector. We want to classify x into correct class C1 or C2 (denoted as a random
variable y). In Logistic Regression, the posterior probability of class C1 can be written as follow:
P(y = C1jx) = (wT x + w0)
where w 2 RD is the weight vector.
For simplicity, we will denote x = [1; x1; x2;    ; xD] and w = [w0;w1;w2;    ;wD]. With this new notation,
the posterior probability of class C1 can be rewritten as follow:
P(y = C1jx) = (wT x) (1)
And posterior probability of class C2 is:
P(y = C2jx) = 1 􀀀 P(y = C1jx)
We now consider the data set fx1; x2;    ; xNg and corresponding label fy1; y2;    ; yNg where
yi =

1 if xi 2 C1
0 if xi 2 C2
for i = 1; 2;    ;N.
With this data set, the likelihood function can be written as follow:
p(yjw) =
NY
n=1
yn
n (1 􀀀 n)1􀀀yn
where n = (wT xn) for n = 1; 2;    ;N.
We also dene the error function by taking the negative logarithm of the log likelihood, which gives the
cross-entropy error function of the form:
E(w) = 􀀀
1
N
ln p(yjw) = 􀀀
1
N
XN
n=1
fyn ln n + (1 􀀀 yn) ln(1 􀀀 n)g (2)
Note that this function is dierent from the squared loss function that we have used for Neural Networks
and Perceptrons.
The gradient of error function with respect to w can be obtained as follow:
rE(w) =
1
N
XN
n=1
(n 􀀀 yn)xn (3)
Up to this point, we can use again gradient descent to nd the optimal weight bw
to minimize the error
function with the formula:
wnew = wold 􀀀 rE(wold) (4)
B Multi-Class Logistic Regression
For multi-class Logistic Regression, the posterior probabilities are given by a softmax transformation of linear
functions of the feature variables, so that
P(y = Ckjx) =
exp(wTk
x) P
j exp(wTj
x)
(5)
4
Now we write down the likelihood function. This is most easily done using the 1-of-K coding scheme in
which the target vector yn for a feature vector xn belonging to class Ck is a binary vector with all elements
zero except for element k, which equals one. The likelihood function is then given by
P(Yjw1;    ;wK) =
NY
n=1
KY
k=1
P(y = Ckjxn)ynk =
NY
n=1
KY
k=1
ynk
nk (6)
where nk is given by (5) and Y is an N  K matrix (obtained using 1-of-K encoding) of target variables
with elements ynk. Taking the negative logarithm then gives
E(w1;    ;wK) = 􀀀ln P(Yjw1;    ;wK) = 􀀀
XN
n=1
XK
k=1
ynk ln nk (7)
which is known as the cross-entropy error function for the multi-class classication problem.
We now take the gradient of the error function with respect to one of the parameter vectors wk . Making
use of the result for the derivatives of the softmax function, we obtain:
@E(w1;    ;wK)
@wk
=
XN
n=1
(nk 􀀀 ynk)xn (8)
then we could use the following updating function to get the optimal parameter vector w iteratively:
wnew
k   wold
k 􀀀 
@E(w1;    ;wK)
@wk
(9)
