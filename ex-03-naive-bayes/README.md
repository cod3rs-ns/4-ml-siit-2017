# Third assignment - Naive Bayes Classifier

## Technologies and libs
- Python 2.7.x
- pandas
- numpy


## Solution description
  We use **Naive Bayes Classifier** algorithm to predict output **contraceptive** for input data
  that contains 8 features.
  
  | Name       | Category   | Values                             |
  |:----------:|:----------:|:----------------------------------:|
  | age        | numeric    |                                    | 
  | wedu       | category   | [1-4]                              |
  | hedu       | category   | [1-4]                              |
  | children   | numeric    |                                    |
  | religion   | binary     | [0, 1]                             |
  | employed   | binary     | [0, 1]                             |
  | standard   | category   | [1-4]                              |
  | media      | binary     | [0, 1]                             |
  
### Continuous features
  We provide options for continuous features and we were able to use **Gaussian probability method**. 
  Also, we were able to provide solution for converting continuous variables to discrete by providing intervals of selection.
  
  Current solution provides **smoothing** with _alpha = 1_ for calculating basic probabilities and no other possibilities.
  We split our _train set_ to 80:20, and we use 20% for **validation** (most common).
  
## Results
  When we applied our algorithm to test set which contains 7 examples, we predict **6** out of 
  **7** correct values. Our **accuracy** is 85.71%.
  
  But, on validation set, our model have **accuracy** of _67.84%_ on 80:20, and _68.04%_ on 70:30 so we can see it's pretty unstable.
  
  On 100:0 we have **accuracy** of 71.42 on test set (5 out of 7). This is default value currently.
  
## Running
  Our solution is very configurable. You can specify percent of validation set, and you can choose which method will be used
  for continuous features. You can combine usage of these solution (ex. make 'age' discrete, but keep 'children' continuous or vice versa).
  Code is well commented so it should not be hard to perform these manipulations.