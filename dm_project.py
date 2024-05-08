import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    X = X.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    X.isna().sum().to_frame().reset_index().rename(columns={0: 'Missing values'})
    X['Age'] = X['Age'].fillna(X['Age'].mean())
    X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
    columns_to_encode = ['Embarked', 'Sex']
    label_encoder = LabelEncoder()
    for column in columns_to_encode:
        X[column] = label_encoder.fit_transform(X[column])
    df_X = X.copy()
    df_X['Survived'] = y
    bins = [0, 9, 14, 42, 57, 59, 100]
    df_X['AgeBin'] = pd.cut(df_X['Age'], bins=bins, labels=[0, 1, 2, 3, 4, 5])
    df_X.groupby(['AgeBin', 'Survived']).size().unstack().apply(lambda x: x / x.sum(), axis=1)
    X['Age'] = X['Age'].apply(lambda age: 0 if age <= bins[1] else 1 if age <= bins[2] else 2 if age <= bins[3] else 3 if age <= bins[4] else 4 if age <= bins[5] else 5)
    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
    X = X.drop(['SibSp', 'Parch'], axis=1)
    X1 = X[['Fare']]
    X2 = X.drop(['Fare'], axis=1)
    scaler = MinMaxScaler().fit(X1)
    X1 = pd.DataFrame(scaler.transform(X1), columns=X1.columns)
    X = pd.concat([X1, X2], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy perceptron: {accuracy}')
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy random forest: {accuracy}')
