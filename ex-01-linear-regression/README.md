# First assignment - Simple linear regression

## Technologies and libs
- Python 2.7.x
- numpy
- matplotlib.pyplot
- csv

## Solution description
 We use simple linear regression `(y_hat = intercept + slope * x)` where `y_hat` is predicted output, `x` is input,
  and `intercept` and `slope` are learned values.

  Main work is done in pre-processing mode. We noticed on graph our data is **exponential**. But, we do little trick -
  on every output value we applied `log` function. Also, before learning coefficients, we normalized our data with
  `(y - median)/std`.

  We removed outliers manually (3 of them that wer out of any sense, 3 with highest value and 3 with the lowest).
  After evaluating our algorithm using **RMSE** method our error is **1.4681**.
  
## Alternative methods

### RANSAC algorithm

  We also tried to fit model with **RANSAC** (Random Sample Consensus) algorithm, because we wanted to use algorithm that
  is resistant to outliers. The main problem in using this algorithm was range of achieved **RMSE** ... **from 1.1 to 9.0**.

##### Why this happens?

  First of all, result of algorithm depends on initial random chosen points. Also, our data distribution (a lot grouped data
  in vertical lines) made that the consensus set is always same, but with different slope and intercept which change
  our model a lot.

### Z-score

 We tried to remove outliers using Z-score. A z-score (aka, a standard score) indicates how many standard deviations an element is from the mean. A z-score can be calculated from the following formula.
 A z-score can be be applied with the following formula (written in Python): `lambda x: np.abs(x - x.mean()) / x.std() < m`. Parameter
 `m` represents what chunk of date based on z-score is selected. Best result which have some meaning is when parameter `m=2` which means we selected about **95%** of elements with best z-score.
 After algorithm is applied with parameter `m=2` on pure data (without manually removed outliers) calculated **RMSE** was equal to **1.2** which is pretty much the best.

##### Note

 We didn't include z-score in code, cause our current RMSE is not that far without z-score. But we provided lambda which can be applied if solution with z-score needs to be tested.