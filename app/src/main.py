# main.py
# ==================================================
# standard
# import os, zipfile, shutil, re, pickle
# from typing import NoReturn
# requirements
# import typer, loguru
# from typer import Option
# defined
from service.service import Analysis
from constants.constants import TICKERS, END_DATE, START_DATE
# --------------------------------------------------

# [INFO] If "data" currently downloaded into a pickle:
# with open('../../temp/data.pkl', 'wb') as file: # for writing object into pkl file.
#     pickle.dump(data, file)
# with open('../../temp/data.pkl', 'rb') as file: # for reading from pkl file into an object.
#     data = pickle.load(file)

# [INFO] Yahoo Finance data
# data = Analysis.yf_download_historical(TICKERS, END_DATE, START_DATE)

# [INFO] If using investing.com csv downloaded files
data = Analysis.read_investing_csv('../../data/investing/')
benchmark_data = Analysis.read_nasdaq_csv('../../data/nasdaq/')

# [INFO] Asset selection
data = data.join(benchmark_data)
returns = Analysis.daily_log_returns(data)
benchmark_returns = returns['SPX']

data = data.drop(columns='SPX')
returns = returns.drop(columns='SPX')

sharpe_ratios = Analysis.annualized_sharpe_ratio(returns)
sortino_ratios = Analysis.annualized_sortino_ratio(returns)

# Jensen’s Alpha
assets_beta = Analysis.estimate_assets_beta(returns, benchmark_returns)

# Calculate Jensen's Alpha for a portfolio or asset
jensen_alpha = Analysis.estimate_jensen_alpha(assets_beta, returns, benchmark_returns)

# Export measures
sharpe_ratios.to_frame().join(sortino_ratios).join(jensen_alpha).reset_index().to_csv('../../temp/measures.csv', index=False)

# [INFO] Weight assignment
top_tickers = ['CDRO', 'ENX', 'GCZ4', 'VOO', 'SMH']
top_returns_data = Analysis.top_tickers_returns(data, top_tickers)

top_mean_returns, _, top_cov_matrix = Analysis.top_tickers_stats(top_returns_data)
top_cov_matrix.reset_index().to_csv('../../temp/cov.csv', index=False)

optimal_weights = Analysis.optimize_portfolio(top_mean_returns, top_cov_matrix)
portfolios_result = Analysis.generate_portfolios(top_mean_returns, top_cov_matrix)
_ = Analysis.plot_efficient_front(portfolios_result)
_ = Analysis.optimal_portfolio_info(optimal_weights, top_mean_returns, top_cov_matrix)
