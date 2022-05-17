import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):

    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    train_X, test_X = np.array(train_X), np.array(test_X)

    # create and fit the model
    adaBoost = AdaBoost(DecisionStump, n_learners)
    adaBoost.fit(train_X, train_y)

    # Question 1: Train- and test errors of AdaBoost in noiseless case

    # get errors
    training_error = np.zeros(n_learners)
    test_error = np.zeros(n_learners)
    for T in range(1, n_learners+1):
        training_error[T-1] = adaBoost.partial_loss(train_X, train_y, T)
        test_error[T-1] = adaBoost.partial_loss(test_X, test_y, T)

    # plot errors
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=np.arange(1, n_learners+1), y=training_error, name='train error', opacity=.75),
                    go.Scatter(x=np.arange(1, n_learners+1), y=test_error, name='test error', opacity=.75)])
    fig.layout.yaxis.title = 'loss'
    fig.layout.xaxis.title = 'number of decision stumps'
    fig.update_layout(title=dict(text='loss as function of number of decision stumps')).show()

    # Question 2: Plotting decision surfaces

    def decision_surface1(T, predict, xrange, yrange, density=120, colorscale=custom):
        """
        function that plots a decision surface
        """
        xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
        xx, yy = np.meshgrid(xrange, yrange)
        pred = predict(np.c_[xx.ravel(), yy.ravel()], T)

        return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=colorscale,
                          reversescale=False, opacity=.7, connectgaps=True, hoverinfo="skip",
                          showlegend=False, showscale=False)

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} learners" for i in T],
                        horizontal_spacing=0.015, vertical_spacing=.035)
    for i, T in enumerate(T):
        fig.add_traces([decision_surface1(T, adaBoost.partial_predict, lims[0], lims[1]),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=test_y.astype(int) + 2,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=1+(i//2), cols=1+(i%2))

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Models With Different Numbers of Learners}}$",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False).show()

    # Question 3: Decision surface of best performing ensemble

    T = np.argmin(test_error) + 1

    # plot the results
    fig = go.Figure()
    fig.add_traces([decision_surface1(T, adaBoost.partial_predict, lims[0], lims[1]),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=test_y.astype(int) + 2,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))
                    ])

    acc = accuracy(adaBoost.partial_predict(test_X, T), test_y)
    fig.update_layout(title=dict(text=f'best classifier. number of classifiers: {T}. accuracy:{acc}'))
    fig.show()

    # Question 4: Decision surface with weighted samples
    D = adaBoost.D_
    D = (D / np.max(D)) * 20
    fig = go.Figure()
    fig.add_traces([decision_surface1(adaBoost.iterations_, adaBoost.partial_predict, lims[0], lims[1]),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                               marker=dict(color=train_y.astype(int), symbol=train_y.astype(int) + 2,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1), size=D))
                    ])

    fig.update_layout(title=dict(text=f'classifier with size proportional to weights')).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
