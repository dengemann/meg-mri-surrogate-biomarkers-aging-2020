"""Utilities for Jupyter Notebook reports."""
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_pred(y, y_pred, mae, title='Prediction vs Measured'):
    """Plot predicted values vs real values."""
    plt.figure()
    plt.title(title)
    plt.scatter(y, y_pred,  edgecolor='black')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '-', lw=3, color='green')
    plt.plot([y.min(), y.max()], [y.min() - mae, y.max() - mae], 'k--', lw=3,
             color='red')
    plt.plot([y.min(), y.max()], [y.min() + mae, y.max() + mae], 'k--', lw=3,
             color='red')
    plt.xlabel('Chronological Age')
    plt.ylabel('Predicted Age')
    plt.grid()
    plt.show()


# https://scikit-learn.org/stable/auto_examples/model_selection/
# plot_learning_curve.html
def plot_learning_curve(train_sizes, train_scores, test_scores,
                        title='Learning Curves', ylim=None):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Numbers of training examples that has been used to generate
        the learning curve.

    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    """
    plt.figure()
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel('Training examples')
    plt.ylabel('Score')

    train_scores = -train_scores
    test_scores = -test_scores

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.legend(loc='best')
    plt.show()


def plot_barchart(mae_std,
                  title='Age Prediction Performance of Different Modalities',
                  bar_text_indent=2):
    """Plot bar chart.

    Parameters
    ----------
    mae_std : dict(str, (number, number))
        Dictionary with labels and corresponding mae and std.
    title : str
        Bar chart title.
    bar_text_indent : number
        Indent from the bar top for labels displaying mae and std,
        measured in years.

    """
    objects = tuple(reversed(sorted(mae_std.keys())))
    y_pos = np.arange(len(objects))
    mae = tuple(mae_std[k][0] for k in objects)
    std = tuple(mae_std[k][1] for k in objects)

    fig, axs = plt.subplots()
    axs.barh(y_pos, mae, align='center', xerr=std)

    # remove frame around the plot
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['left'].set_visible(False)

    for i, v in enumerate(mae):
        axs.text(v + bar_text_indent, i - 0.05,
                 f'{round(v, 2)} ({round(std[i], 2)})',
                 color='blue', bbox=dict(facecolor='white'))

    plt.yticks(y_pos, objects)
    plt.xlabel('Absolute Prediction Error (Years)')
    plt.title(title)
    plt.show()


def plot_boxplot(data, title='Age Prediction Performance'):
    """Plot box plot.

    Parameters
    ----------
    data : dict(str, numpy.ndarray)
        Dictionary with labels and corresponding data.
    title : str
        Bar chart title.

    """
    data_pd = pd.DataFrame(data)
    sns.set_style('darkgrid')
    plt.figure()
    ax = sns.boxplot(data=data_pd, showmeans=True, orient='h')
    ax.set_title(title)
    ax.set(xlabel='Absolute Prediction Error (Years)')
    plt.show()


def plot_error_scatters(data, title='AE Scatter', xlim=None, ylim=None):
    """Plot prediction errors of different modalities versus each other."""
    data = data.dropna()
    age = data.age.values
    color_map = plt.cm.viridis((age - min(age)) / max(age))
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    for key1, key2 in combinations(keys, r=2):
        fig, ax = plt.subplots()
        x_values = np.abs(data[key1].values - age)
        y_values = np.abs(data[key2].values - age)
        plt.scatter(x_values, y_values, edgecolors='black', color=color_map)
        plt.title(title)
        plt.xlabel(key1)
        plt.ylabel(key2)

        if xlim is not None:
            xlim_ = (xlim[0] - 1, xlim[1] + 1)
        else:
            xlim_ = (data[key1].min() - 1, data[key1].max() + 1)

        if ylim is not None:
            ylim_ = (ylim[0] - 1, ylim[1] + 1)
        else:
            ylim_ = (data[key2].min() - 1, data[key2].max() + 1)

        ax.set(xlim=xlim_, ylim=ylim_)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls='--', c='.3')
        plt.grid()


def plot_error_age(data, title='AE vs Age', xlim=None, ylim=None):
    """Plot prediction errors of different modalities versus subject's age."""
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    for key1 in keys:
        data_slice = data[key1].dropna()
        age = data.loc[data_slice.index, 'age'].values
        abs_errors = np.abs(data_slice.values - age)
        plt.figure()
        plt.scatter(age, abs_errors, edgecolors='black')
        plt.title(title)
        plt.xlabel('Age (Years)')
        plt.ylabel(key1)
        plt.grid()

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)


def plot_error_segments(data, segment_len=10, title=None, figsize=None,
                        xlim=(0, 55)):
    """Plot prediction errors for different age groups."""
    keys = data.columns
    # remove column with the original age
    keys = keys[1:]
    age = data.age.values
    for key in keys:
        n_segments = int((age.max() - age.min()) / segment_len)
        segments_dict = {}
        plt_title = 'AE per Segment, %s' % key if title is None else title
        age_pred = data[key]

        for i in range(0, n_segments):
            bound_low = age.min() + i * segment_len
            bound_high = age.min() + (i + 1) * segment_len

            if i == n_segments - 1:
                indices = age >= bound_low
            else:
                indices = (age >= bound_low) * (age < bound_high)

            segments_dict[f'{bound_low}-{bound_high}'] =\
                np.abs(age[indices] - age_pred[indices])

        df = pd.DataFrame.from_dict(segments_dict, orient='index').transpose()

        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df, showmeans=True, orient='h')
        ax.set_title(plt_title)
        ax.set(xlim=xlim, xlabel='Absolute Prediction Error (Years)',
               ylabel='Age Ranges')
        plt.show()
