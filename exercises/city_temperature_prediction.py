import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear #apply(lambda x: pd.Period(x).day_of_year)
    df.drop(columns=["Date"], inplace=True)
    df = df[df.Temp > -70]
    df.dropna(inplace=True)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_il = df[df["Country"] == "Israel"]

    px.scatter(x=df_il.DayOfYear, y=df_il.Temp, color=df_il.Year.astype(str),
               title="Average temperature in Israel as function of the day of year",
               labels=dict(x="day of year", y="average temperature", color="year")).show()

    px.bar(x=df_il.Month.unique(), y=df_il.groupby(['Month'])['Temp'].agg('std'),
           title="Standard deviation of temperature in Israel for each month",
           labels=dict(x="month", y="std of temperature")).show()

    # Question 3 - Exploring differences between countries
    a = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()

    px.line(x=a['Month'], y=a["mean"], color=a["Country"], error_y=a["std"],
            title="Average with CI of temperature by month for different countries",
            labels=dict(x="month", y="average temp", color="country")).show()

    # Question 4 - Fitting model for different values of `k`
    X, y = df_il["DayOfYear"], df_il["Temp"]

    X_train, y_train, X_test, y_test = split_train_test(X, y)
    k_ = np.linspace(1, 10, 10).astype(int)
    loss = np.ones(10)

    for k in k_:
        model = PolynomialFitting(k)
        model.fit(X_train.values, y_train.values)
        loss[k-1] = np.round(model.loss(X_test.to_numpy(), y_test.to_numpy()), 2)
        print(f'mse for k = {k}: {loss[k-1]}')

    px.bar(x=k_, y=loss,
           title="Average MSE of our model as function of polynomial degree",
           labels=dict(x="polynomial degree", y="average MSE")).show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    model.fit(X.to_numpy(), y.to_numpy())
    countries = df["Country"].unique()
    losses = np.zeros_like(countries)

    for i, country in enumerate(countries):
        X_test = df.loc[df["Country"] == country, ["DayOfYear"]]
        y_test = df.loc[df["Country"] == country, ["Temp"]]
        loss = model.loss(X_test.to_numpy().flatten(), y_test.to_numpy().flatten())
        losses[i] = loss

    px.bar(x=countries, y=losses,
           title="Model's MSE over different countries",
           labels=dict(x="country", y="model's MSE")).show()

