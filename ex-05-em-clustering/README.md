# Fifth assignment - Expectation maximization algorithm

## Technologies and libs
- Python 2.7.x
- pandas
- scikit-image

## Solution description
We used **Expectation Maximization algorithm** with **Gaussian probability** with _unknown_ **mean** and **sigma**.

For input features we used **H (hue)**, **S (saturation)** and **V (value)** from each image.

We did not perform any preprocessing on images because our features are already **normalized**.


## Results
On our test set (we added 4 more examples recognized by us) accuracy is 75%, but on 200 examples we checked manually we 
had 103 correct predictions and 97 wrong predictions which means accuracy on larger sample is about 50%.  