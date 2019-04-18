# twitter_sentiment_analysis #
## tools and library ##
Jupyter Notebook, pandas, scikit learn
## Dataset Description ##
Twitter sentiment analysis is a good example of natural language processing. this dataset contains 31000+ data with two classes. **Hate speech** and 
**straight speech** . which was been taken from analytics vidya website.

## Project Description ##
This project predict text of twitter whether it is straight speech or hate speech. Data were preprocessed by trimming digit,puncuation and 
other bad charecter.After that,the dataset are converted from text to numerical tabular data using Countvectorizer ( a scikit learn provided library which convert text data to sparse matrix where each word is used as a token.) **20 %** of 
data were used to test. Comparing **Multonomial Naive bayes** with **Logistic Regression** **Logistic regression** provide better performance
with accuracy **95.97%** where **Multinomial Naive Bayes** provide accuracy **95.35%** accuracy. **GridsearchCV** is used to get best hyper parameter.
