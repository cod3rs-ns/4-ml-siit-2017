# Second assignment - Multi variable linear regression

## Technologies and libs
- Python 2.7.x
- numpy
- sklearn.metrics
- pandas

## Solution description
  We use **K nearest neighbour** algorithm to predict output **grade** for input data
  that contains 26 features.
  
### Features we used

  | Name       | Category   | Values                             |
  |:----------:|:----------:|:----------------------------------:|
  | sex        | binary     | [F, M]                             | 
  | Medu       | numeric    | [0-4]                              |
  | reason     | nominal    | [home, reputation, course, other]  |
  | studytime  | numeric    | [1-4]                              |
  | schoolsup  | binary     | [yes, no]                          |
  | higher     | binary     | [yes, no]                          |
  | goout      | numeric    | [1-5]                              |
  | Dalc       | numeric    | [1-5]                              |
  | Walc       | numeric    | [1-5]                              |
  | Internet   | binary     | [yes, no]                          |
  | Guardian   | nominal    | [mother, father, other]            |

  Main work was based on finding appropriate features and preproccessing data for those features.

### Preprocessing data

  All features from ***numeric*** category are normalized using formula:  
  * (x<sub>i</sub> - median(x)) / (std(x))
  
  We also substitute string values from binary features with 0 and 1, to get numerical representation.  
  
  And finally, from every nominal feature we create **n** (n - number of categories) new features. 
  For example if value of ***reason*** feature for one input vector was ***home***, value is mapped as:
   
  |  ...  |  home   |  reputation  |  course  |  other  |  ... |
  |:-----:|:-------:|:------------:|:--------:|:-------:|:----:|
  |  ...  |    1    |       0      |     0    |    0    |  ... |
  
## Results
  From experimental analysis we achieved best results with **k**=19 (number of neighbours that
  should participate in finding best class for one input vector).
  
  We separate our training data in train and validation set **(80/20)** and evaluate **RMSE**.  
   **RMSE** on validation set was **2.52**.
