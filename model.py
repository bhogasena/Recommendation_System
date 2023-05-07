import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    '''function to get the top product 20 recommendations for the user'''

    def getRecommendationByUser(self, user):
        recommedations = []
        return list(self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index)

    '''function to filter the product recommendations using the sentiment model and get the top 5 recommendations'''

    def getSentimentRecommendations(self, user):
        if (user in self.user_final_rating.index):
            # get the product recommedation using the trained ML model
            recommendations = self.getRecommendationByUser(user)
            temp = self.cleaned_data[self.cleaned_data.id.isin(
                recommendations)]
            # transfor the input data using saved tf-idf vectorizer
            X = self.vectorizer.transform(
                temp["reviews_text_cleaned"].values.astype(str))
            temp_df = temp.copy()
            temp_df.loc[:, "predicted_sentiment"] = self.model.predict(X)
            temp_df = temp_df[['id', 'predicted_sentiment']]
            temp_grouped = temp_df.groupby('id', as_index=False).count()
            temp_grouped_df = temp_grouped.copy()
            temp_grouped_df.loc[:, "pos_review_count"] = temp_grouped_df.id.apply(lambda x: temp_df[(
                temp_df.id == x) & (temp_df.predicted_sentiment == 1)]["predicted_sentiment"].count())
            temp_grouped_df.loc[:,
                                "total_review_count"] = temp_grouped_df['predicted_sentiment']
            temp_grouped_df.loc[:, 'pos_sentiment_percent'] = np.round(
                temp_grouped_df["pos_review_count"]/temp_grouped_df["total_review_count"]*100, 2)
            sorted_products = temp_grouped_df.sort_values(
                'pos_sentiment_percent', ascending=False)[0:5]
            return pd.merge(self.data, sorted_products, on="id")[["name", "brand", "manufacturer", "pos_sentiment_percent"]].drop_duplicates().sort_values(['pos_sentiment_percent', 'name'], ascending=[False, True])

        else:
            print(f"User name {user} doesn't exist")
            return None

    """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    """function to preprocess the text before it's sent to ML model"""

    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        # Keep only word charatcers and white spaces and remove other punctuations marks
        text = re.sub(r'[^\w\s]', '', text)
        # remove words that has digit in it
        text = re.sub("\S*\d\S*", "", text)
        # remove text that has pattern of '[ <text> ]'
        text = re.sub("\[\s*\w*\s*\]", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    '''Function to remove the stop words'''

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag"""

    def lemma_text(self, text):
        wordnet_tags = {'N': 'n', 'V': 'v', 'R': 'r', 'J': 'a'}
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], wordnet_tags.get(tag[1],'n')) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)


if __name__ == '__main__':
    rec = SentimentRecommenderModel()
    rec.getSentimentRecommendations('02dakota')
