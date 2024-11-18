import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer


def transform_data(df, num_cols, cat_cols, age_col):
    le = LabelEncoder()
    scaler = MinMaxScaler()

    cat_cols.remove('Cabin')
    cat_cols.remove('Name')
    cat_cols.remove('Ticket')
    num_cols.remove('PassengerId')

    df = df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

    knn = KNNImputer(n_neighbors=5)
    inputed = knn.fit_transform(df[num_cols])
    df['Age'] = inputed[:, age_col]

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

    df[num_cols] = scaler.fit_transform(df[num_cols])

    for name in cat_cols:
        df[name] = le.fit_transform(df[[name]])

    return df


def transform_data_lab3(df):
    corr_matrix = df.corr()

    top_corr_features = corr_matrix.index[abs(corr_matrix['Цена']) > 0.3]

    train_df = df[top_corr_features]
    features = train_df.drop('Цена', axis=1)
    corr_matrix = features.corr()

    high_corr = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.75:
                high_corr.add((corr_matrix.columns[i], corr_matrix.columns[j]))

    price_corr = train_df.corr()['Цена']
    to_drop = set()
    for feature_1, feature_2 in high_corr:
        if abs(price_corr[feature_1]) < abs(price_corr[feature_2]):
            to_drop.add(feature_1)
        else:
            to_drop.add(feature_2)
    train_df.drop(to_drop, axis=1, inplace=True)

    y = train_df['Цена']
    x = train_df.drop('Цена', axis=1)
    return x, y


