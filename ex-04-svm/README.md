# Fourth assignment - Support Vector Machine

## Technologies and libs
- Python 2.7.x
- pandas
- numpy
- scikit-learn


## Solution description
  We use **SVM (Support Vector Machine)** algorithm to predict output **class** of tumor
  (0-benign or 1-malign) for input data that contains 9 features. We tried all possible
  kernels from scikit-learn library and we assumed best results with **sigmoid** function.
  
  | Name       | Category   | Values                             |
  |:----------:|:----------:|:----------------------------------:|
  | clump      | numeric    | [1,10]                             | 
  | size       | numeric    | [1,10]                             |
  | shape      | numeric    | [1,10]                             |
  | adhesion   | numeric    | [1,10]                             |
  | epithelial | numeric    | [1, 10]                            |
  | nuclei     | numeric    | [1,10]                             |
  | chromatin  | numeric    | [1,10]                             |
  | nucleoli   | numeric    | [1, 10]                            |
  | mitoses    | numeric    | [1, 10]                            |
 
### Data preprocessing 
  We realized that we have a lot of missing data in our train data. We conclude that
  all missing values are values from feature **nuclei**. Because of that we tried to remove
  this feature from our model, but the result was worse. Also we tried to remove only samples
  with missing values, but we also did not improve model. We had best results when we interpolate
  missing values with **mean** value of feature. 
  
  Also we had better results with data normalization using formula:  
  **norm = (x<sub>i</sub> - min) / (max - min)** 

## Results
  When we applied our algorithm to test set which contains ***5 examples***, we achieved **f1-score** to be **1.0**. 
  But, on validation set, our model has f1-score of **0.9906542** on 80:20.

  Other results:
  
  | Description                         |         Result  |
  |:------------------------------------|:---------------:|
  | without 'nuclei' feature            |  0.953271       |
  | without rows with missing values    |  0.961538       |
  
  Also we try formula **norm = (x<sub>i</sub> - mean) / (max - min)** for normalization and we achieved
  same result for training data, but we had worse result on test data (0.8).

## Running
  Our solution is very configurable. You can specify percent of validation set.
  Code is well commented so it should not be hard to perform these manipulations.