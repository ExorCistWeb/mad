import statsmodels.api as sm
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy import stats


def calculate_r_square(y_true, y_pred):
    return 1 - ((y_true - y_pred) ** 2).sum() / (
                (y_true - y_true.mean()) ** 2).sum()


def calculate_adj(y_true, y_pred, x_test):
    r_squared = calculate_r_square(y_true, y_pred)

    n = len(y_true)
    k = x_test.shape[1]

    return 1 - ((1 - r_squared) * (n - 1)) / (n - k - 1)


def calculate_aic(x_test, coef, mse):
    return len(coef) * 2 + len(x_test) * np.log(mse)


def calculate_bic(x_test, coef, mse):
    log_likelihood = -len(x_test) / 2 * (1 + np.log(2 * np.pi * mse))
    return -2 * log_likelihood + len(coef) * np.log(len(x_test))


def calculate_statistics(y_true, y_pred, x_test, coef):
    stats = {}
    mse = mean_squared_error(y_true, y_pred)
    stats['R^2'] = calculate_r_square(y_true, y_pred)
    stats['Adj.R^2'] = calculate_adj(y_true, y_pred, x_test)
    stats['RMSE'] = np.sqrt(mse)
    stats['AIC'] = calculate_aic(x_test, coef, mse)
    stats['BIC'] = calculate_bic(x_test, coef, mse)
    return stats


def cross_validation_calculate_statistics(models, x_train, y_train, x_test,
                                          y_test):
    stats_df = pd.DataFrame(
        {'Name': [], 'R^2': [], 'Adj.R^2': [], 'RMSE': [], 'AIC': [],
         'BIC': []})

    for model_name, data in models.items():
        model = data['model']
        params = data['params']

        grid = GridSearchCV(model, params, scoring='neg_mean_squared_error')
        grid.fit(x_train, y_train)
        y_pred = grid.predict(x_test)
        values = calculate_statistics(y_test, y_pred, x_test, grid.best_estimator_.coef_).values()
        tmp = [model_name, *list(values)]
        stats_df = pd.concat(
            [stats_df, pd.DataFrame([tmp], columns=stats_df.columns)],
            ignore_index=True)

    return stats_df


def calculate_x_statistics(models, x_train, y_train, x_test, y_test):
    for model_name, data in models.items():
        stats_df = pd.DataFrame(
            {'Name': [], 'Estimate': [], 'Standard Error': [], 't-value': [],
             'p-value': [], '95% Confidence Interval': [],
             'Coefficient is significant': []})
        model = data['model']
        params = data['params']

        grid = GridSearchCV(model, params, scoring='neg_mean_squared_error')
        grid.fit(x_train, y_train)
        best = grid.best_estimator_
        y_pred = grid.predict(x_test)

        # Остатки
        residuals = y_test - y_pred

        # Кол-во наблюдений, предикторов и остатков
        n_obs = x_test.shape[0]
        n_pred = x_test.shape[1]
        n_res = n_obs - n_pred - 1

        # Средняя квадратическая ошибка
        mse = np.mean((y_pred - y_test) ** 2)
        mse_res = np.mean(residuals ** 2)

        # Стандартная ошибка
        se = np.sqrt(
            np.diag(np.linalg.inv(np.dot(x_test.T, x_test))) * mse_res)

        t_values = best.coef_ / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), n_res))

        # 95% доверительый интервал
        conf_int = np.zeros((n_pred, 2))
        for i in range(n_pred):
            conf_int[i] = [best.coef_[i] - stats.t.ppf(1 - 0.05/2, n_res) * se[i],
                           best.coef_[i] + stats.t.ppf(1 - 0.05/2, n_res) * se[i]]

        for i in range(n_pred):
            tmp = [x_test.columns[i], best.coef_[i], se[i], t_values[i], p_values[i], conf_int[i], 'Yes' if np.abs(t_values[i]) > 1.96 and p_values[i] < 0.05 else 'No']
            tmp[5] = str(tmp[5])

            stats_df = pd.concat([stats_df, pd.DataFrame([tmp], columns=stats_df.columns)], ignore_index=True)

        yield model_name, stats_df


def sm_grid_search_cv(model, x_train, y_train):
    kf = KFold(n_splits=5, shuffle=True)
    min_rmse = np.inf
    best_model = None
    results = None

    param_grid = {
        'sigma': [1, 2, 3],
        'missing': ['none', 'drop', 'raise'],
        'hasconst': [True, False],
        'weights': ['var_inv', 'unweighted'],
    }
    if model == 'GLS':
        for sigma in param_grid['sigma']:
            for missing in param_grid['missing']:
                for hasconst in param_grid['hasconst']:
                    rmse = 0
                    for train_index, test_index in kf.split(x_train):
                        k_x_train, k_x_test = x_train.iloc[train_index], x_train.iloc[test_index]
                        k_y_train, k_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
                        model = sm.GLS(k_y_train, k_x_train, sigma=sigma, missing=missing, hasconst=hasconst)
                        results = model.fit()
                        y_pred = results.predict(k_x_test)
                        rmse += mean_squared_error(k_y_test, y_pred, squared=False)

                    rmse /= 5
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_model = results
    elif model == 'WLS':
        for missing in param_grid['missing']:
            for hasconst in param_grid['hasconst']:
                rmse = 0
                for train_index, test_index in kf.split(x_train):
                    k_x_train, k_x_test = x_train.iloc[train_index], \
                                          x_train.iloc[test_index]
                    k_y_train, k_y_test = y_train.iloc[train_index], \
                                          y_train.iloc[test_index]
                    model = sm.WLS(k_y_train, k_x_train, missing=missing, hasconst=hasconst)
                    results = model.fit()
                    y_pred = results.predict(k_x_test)
                    rmse += mean_squared_error(k_y_test, y_pred, squared=False)

                rmse /= 5
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_model = results

    return best_model, min_rmse
