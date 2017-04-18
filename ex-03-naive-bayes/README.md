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
  
  We provide options for continuous variables and we were able to use Gaussian probability method. 
  Also, we were able to provide converting discrete variables to intervals, but we must provide boundary.
  
  Current solution provides **smoothing** with _alpha = 1_ for calculating basic probabilities and no other possibilities.
  We split our _train set_ to 80:20, and we use 20% for **validation**.
  
## Results
  When we applied our algorithm to test set which contains 7 examples, we predict **6** out of 
  **7** correct values. Our **accuracy** is 85.71%.
  
  But, on validation set, our model have **accuracy** of 67.84%, so we can see it's pretty unstable.