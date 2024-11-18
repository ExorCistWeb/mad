from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

def transform_data(df, num_cols, cat_cols, age_col):
    le = LabelEncoder()
    scaler = MinMaxScaler()

    cat_cols.remove('Cabin')
    cat_cols.remove('Name')
    cat_cols.remove('Ticket')
    num_cols.remove('PassengerId')
    
    # cat_cols = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    # num_cols = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    # - Embarked
    # - Fare
    # - Parch
    # cat_cols.remove('Embarked')
    # num_cols.remove('Fare')
    # num_cols.remove('Parch')
    df = df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)

    # df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    knn = KNNImputer(n_neighbors=5)
    inputed = knn.fit_transform(df[num_cols])
    # print(inputed[:, age_col])
    df['Age'] = inputed[:, age_col]
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())

    df[num_cols] = scaler.fit_transform(df[num_cols])

    for name in cat_cols:
        df[name] = le.fit_transform(df[[name]])

    return df