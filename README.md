# house-price-regression

Predicting house sale prices with linear regression. First supervised learning project, from Module 3 of the DSML course.

## what it does

- loads 4,600 house sales and cleans them (pulls sale month and year out of the date, drops columns that can't help)
- builds a baseline that always guesses the average price, as the floor to beat
- trains linear regression and ridge regression, tuning ridge's alpha with grid search
- compares all three with 5-fold cross-validation on the training data (ridge re-tuned inside each fold)
- plots predicted vs actual prices and the residuals

## results

- baseline: R2 -0.007 on the test split, essentially zero, since it just guesses the mean
- linear regression: R2 0.567 on the test split; ridge: 0.553 (best alpha 10) - no evidence regularization helps on this data
- in 5-fold cross-validation both models swing from about 0.09 to 0.57 depending on the fold (mean about 0.39), which is why one split isn't enough to judge a model
- the most extreme sales dominate the errors: in one fold, the single priciest house (26.6M) made up 73% of the squared error on its own

## data

`House_Price_data.csv` comes from the instructor's [DSML-Training repo](https://github.com/aamir-gk2539-byte/DSML-Training) (Module 3). The notebook uses the CSV if it's in this folder and downloads it from the repo otherwise.

## how to run

```
pip install -r requirements.txt
jupyter notebook house_price_regression.ipynb
```

Built and tested on Python 3.10.

## limits

- the data is one metro area over a short period, so it won't transfer to other markets
- linear models can't capture interactions (a view may matter more in one city than another). tree-based models would handle that better, and they come later in this module
- the biggest misses are the most expensive houses, which the model underprices
