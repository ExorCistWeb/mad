import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


def plot_grid_search(clf, x_train, y_train, x_test, y_test, param_grid, cv=5):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axes = axes.ravel()

    for i, param in enumerate(param_grid):
        param_values = param_grid[param]
        train_scores_mean = []
        test_scores_mean = []

        for value in param_values:
            local_param_grid = {param: [value]}
            model = GridSearchCV(clf(), local_param_grid, cv=cv)
            model.fit(x_train, y_train)
            if isinstance(clf, DecisionTreeClassifier):
                y_train_pred = model.predict_proba(x_train)[:, 1]
                y_test_pred = model.predict_proba(x_test)[:, 1]
                train_auc = roc_auc_score(y_train, y_train_pred)
                test_auc = roc_auc_score(y_test, y_test_pred)
                # cv_auc = np.mean(model.cv_results_['mean_test_score'])
            else:
                train_auc = model.score(x_train, y_train)
                test_auc = model.score(x_test, y_test)
                # cv_auc = np.mean(model.cv_results_['mean_test_score'])

            train_scores_mean.append(train_auc)
            test_scores_mean.append(test_auc)
            # cv_scores_mean.append(cv_auc)

        axes[i].plot(param_values, train_scores_mean, label='train_auc')
        axes[i].plot(param_values, test_scores_mean, label='test_auc')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel('ROC-AUC')
        # axes[i].plot(param_values, cv_scores_mean, label='cross_val')
        axes[i].set_title(param)
        axes[i].legend()

    fig.tight_layout()
    plt.show()


def draw_super_puper_plot(clf, x_train, y_train, x_test, y_test, max_depth):
    test_scores = [[], [], []]
    max_depth_list = list(range(1, max_depth))
    for i in range(3):
        max_leaf = (i + 1) * 10
        for value in max_depth_list:
            tmp_model = clf(max_leaf_nodes=max_leaf, max_depth=value)
            tmp_model.fit(x_train, y_train)
            if isinstance(clf, DecisionTreeClassifier):
                y_test_pred = tmp_model.predict_proba(x_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_test_pred)
            else:
                test_auc = tmp_model.score(x_test, y_test)
            test_scores[i].append(test_auc)

    plt.plot(max_depth_list, test_scores[0], label='max_leaf_10')
    plt.plot(max_depth_list, test_scores[1], label='max_leaf_20')
    plt.plot(max_depth_list, test_scores[2], label='max_leaf_30')
    plt.legend()
    plt.show()


def roc_auc_plot(y_train, y_train_predicted, y_val, y_val_predicted):
    train_auc = roc_auc_score(y_train, y_train_predicted)
    test_auc = roc_auc_score(y_val, y_val_predicted)

    plt.figure(figsize=(10,7))
    plt.plot(*roc_curve(y_train, y_train_predicted)[:2], label=f'train AUC={train_auc:.4f}')
    plt.plot(*roc_curve(y_val, y_val_predicted)[:2], label=f'test AUC={test_auc:.4f}')
    legend_box = plt.legend(fontsize='large', framealpha=1).get_frame()
    legend_box.set_facecolor("white")
    legend_box.set_edgecolor("black")
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    plt.show()


def draw_rmse_regression(clf, x_train, y_train, x_test, y_test):
    depth = list(range(2, 100))

    rmse = []
    for i in depth:
        local_rmse = []
        for j in range(10):
            model = clf(max_depth=i)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            local_rmse.append(mean_squared_error(y_test, y_pred, squared=True))
        rmse.append(sum(local_rmse) / len(local_rmse))

    # rmse = []
    # depth = []
    # for i in range(2, 100):
    #     model = clf(max_depth=i)
    #     model.fit(x_train, y_train)
    #     y_pred = model.predict(x_test)
    #     depth.append(i)
    #     rmse.append(mean_squared_error(y_test, y_pred, squared=True))

    plt.figure(figsize=(10, 7))
    plt.plot(depth, rmse, label=f'test RMSE: {sum(rmse) / len(rmse)}')
    plt.title('Depth RMSE')
    plt.legend()
    plt.show()


def draw_rmse_regression_leafs(clf, x_train, y_train, x_test, y_test):
    leaf_nodes = list(range(2, 100))

    rmse = []
    for i in leaf_nodes:
        local_rmse = []
        for j in range(10):
            model = clf(max_leaf_nodes=i)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            local_rmse.append(mean_squared_error(y_test, y_pred, squared=True))
        rmse.append(sum(local_rmse) / len(local_rmse))


    # rmse = []
    # depth = []
    # for i in range(2, 100):
    #     model = clf(max_leaf_nodes=i)
    #     model.fit(x_train, y_train)
    #     y_pred = model.predict(x_test)
    #     depth.append(i)
    #     rmse.append(mean_squared_error(y_test, y_pred, squared=True))

    plt.figure(figsize=(10, 7))
    plt.plot(leaf_nodes, rmse, label=f'test RMSE: {sum(rmse) / len(rmse)}')
    plt.title('Max leaf nodes RMSE')
    plt.legend()
    plt.show()


def draw_tree(cl):
    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(cl, filled=True, fontsize=30)
