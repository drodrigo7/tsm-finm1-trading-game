# main.py
# ==================================================
# standard
import os, zipfile, shutil, re, pickle
from typing import NoReturn
# requirements
import typer, loguru
from typer import Option
# defined
from service.service import Analysis
# --------------------------------------------------

# data = Analysis.download_historical()

# [INFO] If data currently downloaded into a pickle:
# with open('./temp/data.pkl', 'wb') as file:
#     pickle.dump(data, file)
with open('./temp/data.pkl', 'rb') as file:
    data = pickle.load(file)

returns = Analysis.daily_returns(data)
sharpe_ratios = Analysis.annualized_stats_sharpe(returns)
top_tickers = Analysis.top_sorted_sharpe(sharpe_ratios)
top_returns = Analysis.top_tickers_returns(data, top_tickers)
top_mean_returns, _, top_cov_matrix = Analysis.top_tickers_stats(top_returns)
optimal_weights = Analysis.optimize_portfolio(top_mean_returns, top_cov_matrix, top_tickers)
portfolios_result = Analysis.generate_portfolios(top_mean_returns, top_cov_matrix)
_ = Analysis.plot_efficient_front(portfolios_result)
_ = Analysis.optimal_portfolio_info(optimal_weights, top_mean_returns, top_cov_matrix)








# app = typer.Typer()
# 
# @app.command(hidden=True)
# def __hidden():
#     ...
# 
# @app.command()
# def download(marker: str = Option(..., help='Asset marker')) -> NoReturn:
#     '''This command downloads information regarding
#     the asset at "marker" parameter.
#     '''
#     
#     logger = loguru.logger
#     logger.info('...')
# 
# @app.command()
# def analize_asset(marker: str = Option(...)) -> NoReturn:
#     '''Analysis of independent asset.
#     '''
#     logger = loguru.logger
#     logger.info('...')
# 
# @app.command()
# def analize_port() -> NoReturn:
#     '''Analisis of complete portfolio.
#     '''
#     logger = loguru.logger
#     logger.info('...')
# 
# if __name__ == '__main__':
#     app()
