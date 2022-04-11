from IMLearn.learners.regressors import LinearRegression
import os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.utils import split_train_test
from IMLearn.metrics import mean_square_error
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pd.read_csv(filename)

    df.drop(columns=["id"], inplace=True)
    df.drop(columns=["date"], inplace=True)
    df.drop(df[df.price <= 0].index, inplace=True)
    df.drop(df[df.sqft_lot15 <= 0].index, inplace=True)

    # df = pd.concat([df, pd.get_dummies(df.zipcode, drop_first=True)], axis=1)
    df.drop(columns=["zipcode"], inplace=True)
    df["total_sqft"] = df["sqft_living"] + df["sqft_lot"] + df["sqft_above"] + df["sqft_basement"]
    df.dropna(inplace=True)
    labels = df["price"]
    features = df.drop(columns=["price"])
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def calculate_pearson_correlation(feature, y):
        return (np.cov(feature, y) / (np.std(feature, ddof=1) * np.std(y, ddof=1)))[0][1]

    corr = pd.DataFrame(data={'feature': X.columns,
                              'corr': np.apply_along_axis(calculate_pearson_correlation, 0, X, y)})

    # "good" feature
    cor = float(corr["corr"][corr["feature"] == "sqft_living"].round(4))
    plot_sqft_living = px.scatter(x=X["sqft_living"], y=y,
                                  title=f"feature: sqft_living    correlation: {cor}",
                                  labels=dict(x="area of living room in sqft", y="price of the house"))

    # "bad" feature
    cor = float(corr["corr"][corr["feature"] == "yr_built"].round(4))
    plot_yr_built = px.scatter(x=X["yr_built"], y=y,
                               title=f"feature: yr_built    correlation: {cor}",
                               labels=dict(x="year the house was built", y="price of the house"))

    # save plots
    pio.write_image(plot_sqft_living, output_path + '/' + "sqft_living.png", format='png')
    pio.write_image(plot_yr_built,  output_path + '/' + "yr_built.png", format='png')


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    model = LinearRegression()
    p_ = np.linspace(0.1, 1, 91).round(2)
    loss = np.zeros_like(p_)
    std = np.zeros_like(p_)
    for j, p in enumerate(p_):
        mse = np.ones(10)
        for i in range(10):
            # sample data
            X_sample = train_X.sample(frac=p)
            y_sample = train_y.loc[X_sample.index]

            # fit model
            model.fit(X_sample.to_numpy(), y_sample.to_numpy())

            # predict test
            # y_pred = model.predict(test_X.to_numpy())
            # mse[i] = mean_square_error(test_y, y_pred.flatten())
            mse[i] = model.loss(test_X.to_numpy(), test_y.to_numpy().flatten())
        loss[j] = mse.mean()
        std[j] = mse.std(ddof=1)

    # fig = go.Figure(go.Scatter(x=p_, y=loss, name="Mean loss"))
    # fig.add_traces([go.Scatter(x=p_, y=loss + 2*std, fill='tonexty', mode="lines", line=dict(color="lightgreen"), name="top of CI"),
    #                 go.Scatter(x=p_, y=loss - 2*std, fill='tonexty', mode="lines", line=dict(color="lightpink"), name="bottom of CI")])
    fig1 = go.Scatter(x=p_, y=loss, name="Mean loss")
    fig2 = go.Scatter(x=p_, y=loss + 2*std, fill='tonexty', mode="lines", line=dict(color="lightgreen"), name="top of CI")
    fig3 = go.Scatter(x=p_, y=loss - 2*std, fill='tonexty', mode="lines", line=dict(color="lightgreen"), name="top of CI")
    a = go.Figure(fig1)
    a.add_traces([fig2, fig3])
    a.show()


    # fig.update_layout(title="Mean loss as a function size of sample",
    #                   xaxis_title="proportion of sample used from available data",
    #                   yaxis_title="loss of model",
    #                   legend_title=None)
    # fig.show()

