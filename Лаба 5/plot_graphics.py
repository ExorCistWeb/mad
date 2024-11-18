import matplotlib.pyplot as plt
import numpy as np


def _plot(train_preds, test_preds, model_name, param, figsize=(7, 7)):
    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, 100), train_preds, label='train')
    plt.plot(np.arange(1, 100), test_preds, label='test')
    plt.xlabel(param)
    plt.ylabel("Score")
    plt.title(model_name)
    plt.legend()
    plt.show()


def plot_model_bagging(ensemble_models, param, models_names, rows, cols,
                       x_train, y_train, x_test, y_test, figsize=(11, 5)):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)

    for i, ensemble_model in enumerate(ensemble_models):
        train_preds = []
        test_preds = []
        for j in range(1, 100):
            match param:
                case "n_estimators":
                    bdt = ensemble_model(n_estimators=j)
                case "max_samples":
                    bdt = ensemble_model(max_samples=j)
                case _:
                    bdt = ensemble_model(max_depth=j)

            bdt.fit(x_train, y_train)

            train_preds.append(bdt.score(x_train, y_train))
            test_preds.append(bdt.score(x_test, y_test))
        axes[i].plot(np.arange(1, 100), train_preds, label='train')
        axes[i].plot(np.arange(1, 100), test_preds, label='test')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Score")
        axes[i].set_title(models_names[i])
        axes[i].legend()

    fig.tight_layout()
    plt.show()


def plot_models(ensemble_model, sub_models, sub_models_names,
                x_train, y_train, x_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes = axes.ravel()

    for i, sub_model in enumerate(sub_models):
        train_preds = []
        test_preds = []
        for j in range(1, 100):
            bdt = ensemble_model(sub_model(), n_estimators=j, random_state=42)

            bdt.fit(x_train, y_train)

            train_preds.append(bdt.score(x_train, y_train))
            test_preds.append(bdt.score(x_test, y_test))

        axes[i].plot(np.arange(1, 100), train_preds, label='train')
        axes[i].plot(np.arange(1, 100), test_preds, label='test')
        axes[i].set_xlabel("n_estimators")
        axes[i].set_ylabel("Score")
        axes[i].set_title(f'{sub_models_names[i]}')
        axes[i].legend()

    fig.tight_layout()
    plt.show()


    # _plot(train_preds, test_preds, model_name, 'max_depth')


def plot_max_depth(ensemble_model, model_name, x_train, y_train, x_test, y_test):
    train_preds = []
    test_preds = []
    for i in range(1, 100):
        bdt = ensemble_model(max_depth=i)

        bdt.fit(x_train, y_train)

        train_preds.append(bdt.score(x_train, y_train))
        test_preds.append(bdt.score(x_test, y_test))

    _plot(train_preds, test_preds, model_name, 'max_depth')


def plot_max_estimators(ensemble_model, model_name, x_train, y_train, x_test, y_test):
    train_preds = []
    test_preds = []
    for i in range(1, 100):
        bdt = ensemble_model(n_estimators=i)

        bdt.fit(x_train, y_train)

        train_preds.append(bdt.score(x_train, y_train))
        test_preds.append(bdt.score(x_test, y_test))

    _plot(train_preds, test_preds, model_name, 'n_estimators')


def plot_max_samples(ensemble_model, model_name, x_train, y_train, x_test, y_test):
    train_preds = []
    test_preds = []
    for i in range(1, 100):
        bdt = ensemble_model(max_samples=i)

        bdt.fit(x_train, y_train)

        train_preds.append(bdt.score(x_train, y_train))
        test_preds.append(bdt.score(x_test, y_test))

    _plot(train_preds, test_preds, model_name, 'max_samples')

# def plot_bagging(ensemble_models, param, models_names, rows, cols,
#                           x_train, y_train, x_test, y_test):
#     fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(11, 5))
#     axes = axes.ravel()
#
#     for i, ensemble_model in enumerate(ensemble_models):
#         train_preds = []
#         test_preds = []
#         for j in range(1, 100):
#             match param:
#                 case "n_estimators":
#                     bdt = ensemble_model(estimator(), n_estimators=j)
#                 case _:
#                     bdt = ensemble_model(estimator(), max_samples=j)
#
#             bdt.fit(x_train, y_train)
#
#             train_preds.append(bdt.score(x_train, y_train))
#             test_preds.append(bdt.score(x_test, y_test))
#         name = "Количество оценщиков" if param == "n_estimators" else "Количество выборок"
#         axes[i].plot(np.arange(1, 100), train_preds, label='train')
#         axes[i].plot(np.arange(1, 100), test_preds, label='test')
#         axes[i].set_xlabel(name)
#         axes[i].set_ylabel("Score")
#         axes[i].set_title(models_names[i])
#         axes[i].legend()
#
#     fig.tight_layout()
#     plt.show()


def plot_model_busting(busting_models, models_names, rows, cols,
                       param, x_train, y_train, x_test, y_test, figsize=(10, 10)):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    axes = axes.ravel()

    for i, busting_model in enumerate(busting_models):
        train_preds = []
        test_preds = []
        for j in range(1, 100):
            match param:
                case "n_estimators":
                    bdt = busting_model(n_estimators=j)
                case _:
                    bdt = busting_model(max_depth=j)

            bdt.fit(x_train, y_train)

            train_preds.append(bdt.score(x_train, y_train))
            test_preds.append(bdt.score(x_test, y_test))

        name = "Количество оценщиков" if param == "n_estimators" else "Глубина"
        axes[i].plot(np.arange(1, 100), train_preds, label='train')
        axes[i].plot(np.arange(1, 100), test_preds, label='test')
        axes[i].set_xlabel(name)
        axes[i].set_ylabel("Score")
        axes[i].set_title(models_names[i])
        axes[i].legend()

    fig.tight_layout()
    plt.show()
