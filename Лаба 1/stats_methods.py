import pandas as pd
from scipy.stats import sigmaclip


def quartile_method(df: pd.Series, column: str) -> pd.Series:
    '''
    Функция метода квартилей

    :param df: Датафрейм
    :param column: Название столбца
    :return: Серия, в которую не попадают выбросы
    '''
    q25, q75 = df[column].quantile([0.25, 0.75])
    iqr = q75 - q25
    low = q25 - 1.5 * iqr
    high = q75 + 1.5 * iqr

    return ((df[column] < low) | (df[column] > high))


def sigma_method(df: pd.Series) -> pd.DataFrame:
    '''
    Функция методы трёх сигм

    :param df: Датафрейм
    :return: Серия, в которую не попадают выбросы
    '''
    num_cols = df.select_dtypes(include=['float']).columns
    for col in num_cols:
        data = df[col].dropna()
        clean_data, low, high = sigmaclip(data, low=3, high=3)
        df = df.loc[(df[col].isin(clean_data)) | (df[col].isna())]

    df = df.reset_index()
    df.pop('index')
    return df
    
    