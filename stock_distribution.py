# -------------------------------------------------------------------------------
# 株価の変化率の分布を調べる
# -------------------------------------------------------------------------------
import warnings
import math
import os
import sys
import datetime
import numpy as np
from scipy import stats
import pandas as pd
import pandas_datareader
import matplotlib.pyplot as plt

START_DATE = datetime.date(2011, 1, 1)
END_DATE = datetime.date(2020, 12, 31)


def get_stock(ticker, start_date, end_date):
    dirname = "data"
    os.makedirs(dirname, exist_ok=True)
    fname = f"{dirname}/{ticker}.pkl"
    df_stock = pd.DataFrame()
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
        start_date = df_stock.index.max() + datetime.timedelta(days=1)
    if end_date > start_date:
        df = pandas_datareader.data.DataReader(
            ticker, "yahoo", start_date, end_date)
        df_stock = pd.concat([df_stock, df[~df.index.isin(df_stock.index)]])
        df_stock.to_pickle(fname)
    return df_stock


def best_fit_distribution(data, bins=50, ax=None):
    """Model data by finding best fit distribution to data"""
    # データからヒストグラムを作成する
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # 以下の分布でフィッティングする
    DISTRIBUTIONS = [
        stats.alpha,
        stats.anglit,
        stats.arcsine,
        stats.beta,
        stats.betaprime,
        stats.bradford,
        stats.burr,
        stats.cauchy,
        stats.chi,
        stats.chi2,
        stats.cosine,
        stats.dgamma,
        stats.dweibull,
        stats.erlang,
        stats.expon,
        stats.exponnorm,
        stats.exponweib,
        stats.exponpow,
        stats.f,
        stats.fatiguelife,
        stats.fisk,
        stats.foldcauchy,
        stats.foldnorm,
        stats.frechet_r,
        stats.frechet_l,
        stats.genlogistic,
        stats.genpareto,
        stats.gennorm,
        stats.genexpon,
        stats.genextreme,
        stats.gausshyper,
        stats.gamma,
        stats.gengamma,
        stats.genhalflogistic,
        stats.gilbrat,
        stats.gompertz,
        stats.gumbel_r,
        stats.gumbel_l,
        stats.halfcauchy,
        stats.halflogistic,
        stats.halfnorm,
        stats.halfgennorm,
        stats.hypsecant,
        stats.invgamma,
        stats.invgauss,
        stats.invweibull,
        stats.johnsonsb,
        stats.johnsonsu,
        stats.ksone,
        stats.kstwobign,
        stats.laplace,
        stats.levy,
        stats.levy_l,
        # stats.levy_stable,  ← 計算が終わらないのでコメントアウト
        stats.logistic,
        stats.loggamma,
        stats.loglaplace,
        stats.lognorm,
        stats.lomax,
        stats.maxwell,
        stats.mielke,
        stats.nakagami,
        stats.ncx2,
        stats.ncf,
        stats.nct,
        stats.norm,
        stats.pareto,
        stats.pearson3,
        stats.powerlaw,
        stats.powerlognorm,
        stats.powernorm,
        stats.rdist,
        stats.reciprocal,
        stats.rayleigh,
        stats.rice,
        stats.recipinvgauss,
        stats.semicircular,
        stats.t,
        stats.triang,
        stats.truncexpon,
        stats.truncnorm,
        stats.tukeylambda,
        stats.uniform,
        stats.vonmises,
        stats.vonmises_line,
        stats.wald,
        stats.weibull_min,
        stats.weibull_max,
        stats.wrapcauchy,
    ]

    # [分布名,分布パラメータ,誤差]を格納するためのリスト
    l = []

    # 分布毎にそれぞれパラメータを推測してみる
    for i, distribution in enumerate(DISTRIBUTIONS):
        print(f"{i+1}/{len(DISTRIBUTIONS)} {distribution.name: <20}", end="")

        # 分布によってはフィットできないこともあるので、
        # フィットできなければ、passする
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # 分布をフィットさせる
                params = distribution.fit(data)

                # わかりやすい用にパラメータをばらばらにする
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # フィットさせた確率密度関数を計算して、分布にフィットさせる
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

                # 残差平方和を誤差として計算する
                sse = np.sum(np.power(y - pdf, 2.0))

                print(f"done sse={sse}")

                # l
                l.append([
                    distribution,
                    params,
                    sse,
                ])
        except Exception:
            print("not fit")

    l.sort(key=lambda x: x[2])
    return l


def make_pdf(distribution, params, start, end, size=500):
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x = np.linspace(start, end, size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    return x, y


def make_graph(fit_list, ticker, x, bins=50):
    COL_NUM = 4
    fig, axes = plt.subplots(math.ceil(len(fit_list) / COL_NUM), COL_NUM,
                             figsize=(10, 30), sharex=True, sharey=True, dpi=300)
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.5)
    plt.title(ticker)
    for i, fit in enumerate(fit_list):
        distribution, params, sse = fit
        x_fit, y_fit = make_pdf(distribution, params, min(x), max(x), bins)
        ax, ay = divmod(i, COL_NUM)
        axes[ax, ay].hist(x, bins=bins, density=True)
        axes[ax, ay].plot(x_fit, y_fit)
        axes[ax, ay].set_title(f"No.{i+1} {distribution.name}")
    plt.savefig(f"stock_distribution_{ticker}.png")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {os.path.basename(sys.argv[0])} stock code...")
        sys.exit(255)

    for ticker in sys.argv[1:]:
        df = get_stock(ticker, START_DATE, END_DATE)
        df["pct_change"] = df["Adj Close"].pct_change()
        x = df["pct_change"].dropna().values
        fit_list = best_fit_distribution(x)

        # 結果の表示
        for i, fit in enumerate(fit_list[0:56]):
            print(f"{i+1} {fit[0].name}")

        # 結果のグラフを作成
        make_graph(fit_list[0:56], ticker, x)


if __name__ == "__main__":
    main()
