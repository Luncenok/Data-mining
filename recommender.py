# Your goal is to build a rule based and clustering based recommender system:
#     - the recommender system based on a user ID and a movie ID calculates a rating for that pair
#     - a pair of designated user ID and a movie ID is not apparent in the training data
#     - the system calculates and denormalizes the rating
#     - the system is limitted to any pre-processing techiniques, rules based methods (e.g assiociataion rule discovery) and clustering methods, specifically the system should not use neural networks for regression
#     - dataset: movies.csv, ratings.csv

# Importing the libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocess_data():
    # Load the dataset
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')

    # Drop unnecessary columns
    ratings = ratings.drop('timestamp', axis=1)

    # Merge the datasets
    data = pd.merge(ratings, movies, on='movieId')

    # genres are in this format: Adventure|Animation|Children|Comedy|Fantasy
    data['genres'] = data['genres'].str.split('|')

    # Encode the 'genre' column using One-Hot Encoding
    data = data.join(data.pop('genres').str.join('|').str.get_dummies())

    # Get year from title
    data['year'] = data['title'].str.extract(r'\((\d{4})\)')
    # 11737       Fawlty Towers (1975-1979)
    # 11823                 Stranger Things
    # 36618       Fawlty Towers (1975-1979)
    # 37669    Big Bang Theory, The (2007-)
    # 80816       The Lovers and the Despot
    # 92909       Fawlty Towers (1975-1979)
    # 95257                      Hyena Road
    # fix above cases
    data.loc[data['title'] == 'Fawlty Towers (1975-1979)', 'year'] = 1977
    data.loc[data['title'] == 'Big Bang Theory, The (2007-)', 'year'] = 2007
    data.loc[data['title'] == 'Stranger Things', 'year'] = 2016
    data.loc[data['title'] == 'The Lovers and the Despot', 'year'] = 2016
    data.loc[data['title'] == 'Hyena Road', 'year'] = 2015

    # Drop 'title' column
    data = data.drop('title', axis=1)

    # one hot encode 'year'
    data = pd.get_dummies(data, columns=['year'], drop_first=True)

    # Drop 'movieId' column
    data = data.drop('movieId', axis=1)

    # rating / 5
    data['rating'] = data['rating'] / 5

    # Save the fully preprocessed DataFrame to a CSV file
    data.to_csv('movies_preprocessed.csv', index=False)

    return data


def knn_classifier(data):
    # Drop unnecessary columns
    X = data.drop(['rating', 'userId'], axis=1)

    # Extract 'rating' column
    y = data['rating']

    # Split the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Perform classification using knn classifier
    from sklearn.neighbors import KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict the Test set results
    y_pred = knn.predict(X_test)

    # Evaluate the model MSE
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return knn



data = preprocess_data()
# data = pd.read_csv('movies_preprocessed.csv')
print(data.head())
print(data.columns)
knn_classifier(data)




