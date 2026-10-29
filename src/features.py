from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

COLS_CATEGORICAS = ["item_condition_id", "shipping"]


class ConstructorFeatures:
    """Convierte el DataFrame en la matriz que entiende el modelo."""

    def __init__(self, max_features=50_000):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3)
        self.ohe = OneHotEncoder(handle_unknown="ignore")

    def fit_transform(self, df):
        return hstack([
            self.tfidf.fit_transform(df["texto"]),
            self.ohe.fit_transform(df[COLS_CATEGORICAS]),
        ]).tocsr()

    def transform(self, df):
        return hstack([
            self.tfidf.transform(df["texto"]),
            self.ohe.transform(df[COLS_CATEGORICAS]),
        ]).tocsr()