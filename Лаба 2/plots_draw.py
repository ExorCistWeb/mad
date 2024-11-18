import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from typing import NoReturn


def draw_cat(serie: pd.Series, title: str, x_name: str,
                     y_name: str) -> NoReturn:
    """
    Функция для отрисовки категориальных признаков.

    :param serie: Серия для отрисовки
    :param title: Название графиков
    :param x_name: Название графиков по оси x
    :param y_name: Название графиков по оси y
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True,
                                   figsize=(10, 4))
    fig.suptitle(title, fontsize=20)

    tdf = serie.value_counts()

    ax1.bar(tdf.keys().to_list(), tdf.to_list())
    ax1.set_xticks(np.arange(len(tdf.keys().to_list())))
    ax1.set_xticklabels(tdf.keys().to_list(), rotation=45, ha='right')
    ax1.set_facecolor('seashell')
    ax1.set_ylabel(y_name)
    ax1.set_xlabel(x_name)

    ax2.pie(tdf.to_list(), labels=tdf.keys().to_list(), autopct='%1.2f%%',
            labeldistance=None, pctdistance=1.3)
    ax2.axis('equal')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

    plt.show()


def draw_num(serie: pd.Series, title: str, x_name: str) -> NoReturn:
    """
    Функция для отрисовки количественных признаков

    :param serie: Серия для отрисовки
    :param title: Название графиков
    :param x_name: Название графиков по оси x
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True,
                                        figsize=(12, 5))
    fig.suptitle(title, fontsize=20)

    ax1.set_ylabel('Плотность относительных частот')
    ax1.set_xlabel(x_name)

    ax2.set_ylabel('Относительные частоты')
    ax2.set_xlabel(x_name)

    ax3.set_xlabel('Значения')

    ax1.hist(serie)

    sns.kdeplot(data=serie, bw_method=0.5, ax=ax2)

    # ax3.hist(column_3)
    sns.boxplot(data=serie, ax=ax3, orient='h')

    # plt.subplots_adjust(wspace=0.8, hspace=0.8)
    plt.show()
