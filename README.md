# house-price-regression

Predicting house sale prices with linear regression. First supervised learning project, from Module 3 of the DSML course.

## what it does

- loads 4,600 house sales and cleans them (pulls sale month and year out of the date, drops columns that can't help)
- builds a baseline that always guesses the average price, as the floor to beat
- trains linear regression and ridge regression, tuning ridge's alpha with grid search
- compares all three with 5-fold cross-validation on the training data (ridge re-tuned inside each fold)
- plots predicted vs actual prices and the residuals
