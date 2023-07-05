import pandas as pd
from numpy import sqrt, abs, round

from scipy.stats import norm
from scipy.stats import t as t_dist

import matplotlib.pyplot as plt
import seaborn as sns


def TwoSampleZTest(X1, X2, sigma1, sigma2, N1, N2):
    """
    takes mean, standard deviation, and number of observations
     and returns p-value calculated for 2-sampled Z-Test
    """

    ovr_sigma = sqrt(sigma1 ** 2 / N1 + sigma2 ** 2 / N2)
    z = (X1 - X2) / ovr_sigma
    p_val = 2 * (1 - norm.cdf(abs(z)))

    return p_val


def TwoSampleTTest(X1, X2, sd1, sd2, n1, n2):
    """
    takes mean, standard deviation, and number of observations
     and returns p-value calculated for 2-sample T-Test
    """

    ovr_sd = sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2)
    t = (X1 - X2) / ovr_sd
    df = n1 + n2 - 2
    p_val = 2 * (1 - t_dist.cdf(abs(t), df))

    return p_val


def Bivariate_plot(data, cont, cat, category):
    """
    Plotting bivariate relationship by creating two samples
    This also performs the hypothesis testing as we go along plotting the graphs.
    """
    # creating 2 samples
    x1 = data[cont][data[cat] == category][:]
    x2 = data[cont][~(data[cat] == category)][:]

    # calculating descriptives
    n1, n2 = x1.shape[0], x2.shape[0]
    m1, m2 = x1.mean(), x2.mean()
    std1, std2 = x1.std(), x2.mean()

    # calculating p-values
    t_p_val = TwoSampleTTest(m1, m2, std1, std2, n1, n2)
    z_p_val = TwoSampleZTest(m1, m2, std1, std2, n1, n2)

    # table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc=np.mean)

    # plotting
    plt.figure(figsize=(20, 4), dpi=140)

    # barplot
    plt.subplot(1, 3, 1)
    sns.barplot([str(category), 'not {}'.format(category)], [m1, m2])
    plt.ylabel('mean {}'.format(cont))
    plt.xlabel(cat)
    plt.title('t-test p-value = {} \n z-test p-value = {}\n {}'.format(t_p_val,
                                                                       z_p_val,
                                                                       table))

    # category-wise distribution
    plt.subplot(1, 3, 2)
    sns.kdeplot(x1, shade=True, color='blue', label='Subscribed')
    sns.kdeplot(x2, shade=False, color='green', label='not Subscribed')
    plt.title('Categorical distribution')
    plt.legend()

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('Categorical boxplot')
