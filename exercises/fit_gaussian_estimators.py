from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model

    # create sample
    sample_size = 1000
    mu = 10
    sigma = 1
    X = np.random.normal(loc=mu, scale=sigma, size=sample_size)

    # create and fit the model
    uni = UnivariateGaussian()
    uni.fit(X)

    print("Estimations for the Univariate parameters:")
    print(f"({uni.mu_}, {uni.var_})\n")

    # Question 2 - Empirically showing sample mean is consistent

    # calculate diff from mu
    estimation_err = []
    for i in range(1, sample_size, 10):
        uni.fit(X[:i])
        estimation_err.append(abs(uni.mu_ - mu))

    # plot the result
    x_len = np.linspace(0, sample_size, sample_size // 10).astype(np.int32)
    go.Figure([go.Scatter(x=x_len, y=estimation_err, mode='markers+lines')],
              layout=go.Layout(
                  title="Distance of Estimation of Expectation from Mu As Function Of Number Of Samples",
                  xaxis=dict(title="number of samples"),
                  yaxis=dict(title="distance of estimator from mu"),
                  height=350)).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    # calculate pdfs
    pdfs = uni.pdf(X)

    # plot results
    go.Figure([go.Scatter(x=X, y=pdfs, mode='markers')],
              layout=go.Layout(
                  title="Samples from ~N(10, 1) distribution and their PDFs",
                  xaxis=dict(title="value of sample"),
                  yaxis=dict(title="PDF of sample"),
                  height=350)).show( )


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    # draw samples
    sample_size = 1000
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=sample_size)

    # create and fit model
    multi = MultivariateGaussian()
    multi.fit(X)

    print("Estimations for our multivariate parameters:\n")
    print("Estimated mu:")
    print(multi.mu_)
    print("\nEstimated cov:")
    print(multi.cov_)

    # Question 5 - Likelihood evaluation

    # calculate Likelihood for all values requested
    res = np.zeros((200, 200))
    f1 = f3 = np.linspace(-10, 10, 200)

    for i in range(200):
        for j in range(200):
            res[i][j] = multi.log_likelihood(np.array([f1[i], 0, f3[j], 0]),
                                             cov,
                                             X)

    # plot heat map of the Likelihoods calculated
    x = np.linspace(-10, 10, 200)
    figure = go.Figure(px.imshow(res, x=x, y=x,
                                 title="log-likelihood of the sample given mu was [f1, 0, f3, 0]",
                                 labels=dict(x="value of f3", y="value of f1", color="log-likelihood"),
                                 color_continuous_scale="BlueRed"))
    figure.add_trace(
        go.Contour(dict(z=res, x=x, y=x,
                        contours=dict(start=res.min(), end=res.max(), size=1000, coloring='lines'),
                        showscale=False)))
    figure.show()

    # Question 6 - Maximum likelihood

    # retrieve index of the maximum value of likelihood
    indices = np.nonzero(res == res.max())

    f1, f3 = round(f1[indices[0][0]], 3), round(f3[indices[1][0]], 3)
    print(f"\nBest model was model with:\nf1: {f1}\nf3: {f3}")

    multi.pdf(X)

if __name__ == '__main__':

    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
