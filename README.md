# Data Analytics Project 1

This project involves analyzing and predicting sentiment values from a set of textual data. The datasets originate from the paper "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank" by Richard Socher et al. The project focuses on processing this data and employing various models to predict sentiment values.

:no_entry_sign: It is not intended for reuse.

:heart: Special thanks to my team [Abuaf Pelo Ian](https://www.linkedin.com/in/ian-abuaf-pelo-70602a2b6/) for his contributions.

## Data Sources
The project uses the following datasets:
1. `original_rt_snippets.txt`: Contains 10,605 processed snippets from Rotten Tomatoes HTML files.
2. `datasetSentences.txt`: Includes sentence index and the corresponding sentence string.
3. `dictionary.txt`: Contains phrases and their IDs.
4. `sentiment_labels.txt`: Includes phrase IDs and their sentiment labels.
5. `SOStr.txt` and `STree.txt`: Encode the structure of the parse trees.
6. `datasetSplit.txt`: Contains sentence index and set label for train, test, and dev sets.

## Data Mapping
Due to the absence of the original Matlab code, various methods were applied to map the original snippets to sentences and phrases, including using word tokenizers and handling inconsistencies manually. The combined dataset is saved in `temp/combined.csv`.

## Data Visualization
Visualizations were created using pandas and seaborn, with additional preprocessing in Python to prepare the data for Tableau. Key visualizations include:
1. Word frequency colored by average sentiment value.
2. Histograms and line plots showing the relationship between sentiment values and phrase lengths.

## Prediction Models
Three models were implemented to predict sentiment values:
1. **Linear Classification**: Using Stochastic Gradient Descent, this model achieved an accuracy of 0.513 and a mean squared error (MSE) of 0.770.
2. **Linear Regression**: Using a linear regressor from the `LinearRegression()` class, this model achieved a score of 0.565 and an MSE of 0.020.
3. **Neural Network Regression**: Using the `MLPRegressor()` class, this model with 4 hidden layers of 100 nodes each, achieved a score of 0.688 and an MSE of 0.014.

## Filtering Dataset
A significant portion of the dataset had phrases with a 0.5 sentiment value, which skewed the model's performance. To address this, the dataset was filtered to remove many neutral phrases using the following criteria:
```python
filter = (Length >= 30) & ((isSentence == True) | (sentimentValue != 0.5))
```

## Feature Extraction
All models preprocess the input with a simple `CountVectorizer()`, converting the strings into their bag-of-words representation. The use of a tf-idf transformer was tested but resulted in worse performance.

## Conclusion
The neural network regressor outperformed the other models, demonstrating the potential for more complex models to better capture the nuances of sentiment in textual data.

