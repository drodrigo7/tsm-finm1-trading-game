# main.py
# ==================================================
# standard
from typing import Any, Collection, Tuple, List
# requirements
import yfinance as yf
import numpy as np
import pandas
# defined
from constants.constants import (
    TICKERS, END_DATE, START_DATE, RISK_FREE_RATE, TRADING_DAYS_YEAR
)
from utils.optimization import Optimization
# --------------------------------------------------

class Analysis(object):
    
    @staticmethod
    def download_historical() -> pandas.DataFrame:
        '''Download historical data'''
        data = yf.download(TICKERS, start=START_DATE, end=END_DATE)['Adj Close']
        # with open('./temp/{}'.format(n), 'rb') as file:
            # data = pickle.load(file)
        return data
    
    @staticmethod
    def daily_returns(data: pandas.DataFrame) -> Any:
        '''Daily returns'''
        returns = np.log(data).diff().dropna()
        return returns
    
    @staticmethod
    def annualized_stats_sharpe(returns: Any) -> Collection:
        mean_returns = np.exp(returns.mean() * TRADING_DAYS_YEAR) - 1
        volatility = np.sqrt(returns.var(ddof=0) * TRADING_DAYS_YEAR)
        sharpe_ratios = (mean_returns - RISK_FREE_RATE) / volatility
        return sharpe_ratios
    
    @staticmethod
    def annualized_stats_sortino() -> Collection:
        sortino_ratios = ...
        return sortino_ratios
    
    @staticmethod
    def top_sorted_sharpe(sharpe_ratios: Collection) -> Any:
        '''Sort by Sharpe ratio (from highest to lowest)'''
        sorted_sharpe = sharpe_ratios.sort_values(ascending=False)
        top_5_tickers = sorted_sharpe.head(5)
        return top_5_tickers
    
    @staticmethod
    def top_sorted_sortino(sortino_ratios: Collection) -> Any:
        '''Sort by Sortino ratio (from highest to lowest)'''
        sorted_sortino = sortino_ratios.sort_values(ascending=False)
        top_5_tickers = sorted_sortino.head(5)
        print('Top 5 tickers based on Sortino Ratio:', top_5_tickers)
        return top_5_tickers
    
    @staticmethod
    def top_tickers_returns(data: pandas.DataFrame, top_tickers: Any) -> Collection:
        '''Top tickers data'''
        top_data = data[top_tickers.index.to_list()]
        top_returns = np.log(top_data).diff().dropna()
        return top_returns
    
    @staticmethod
    def top_tickers_stats(top_returns: Any) -> Tuple[Any]:
        '''Calculate mean returns and covariance matrix'''
        top_mean_returns = np.exp(top_returns.mean() * TRADING_DAYS_YEAR) - 1
        top_volatility = np.sqrt(top_returns.var(ddof=0) * TRADING_DAYS_YEAR)
        top_cov_matrix = top_returns.cov() * 252
        return top_mean_returns, top_volatility, top_cov_matrix
    
    @staticmethod
    def optimize_portfolio(top_mean_returns: Any, top_cov_matrix: Any, top_tickers: List[str]) -> Any:
        '''Optimize portfolio'''
        optimal_result = Optimization.optimize_portfolio(top_mean_returns, top_cov_matrix, top_tickers, RISK_FREE_RATE)
        optimal_weights = optimal_result.x
        return optimal_weights
    
    @staticmethod
    def generate_portfolios(top_mean_returns: Any, top_cov_matrix: Any) -> Any:
        '''Generate portfolios and visualize'''
        num_portfolios = 10000
        portfolios_result, _ = Optimization.generate_portfolios(num_portfolios, top_mean_returns, top_cov_matrix, RISK_FREE_RATE)
        return portfolios_result
    
    @staticmethod
    def plot_efficient_front(portfolios_result: Collection) -> None:
        '''Plot Efficient Frontier'''
        _ = Optimization.efficient_frontier_plot(portfolios_result)
        return None
    
    @staticmethod
    def optimal_portfolio_info(optimal_weights, top_mean_returns, top_cov_matrix) -> None:
        '''Analyze the Optimal Portfolio'''
        opt_return, opt_std = Optimization.portfolio_performance(optimal_weights, top_mean_returns, top_cov_matrix)
        print("Optimal Portfolio Weights:", optimal_weights)
        print("Expected Portfolio Return:", opt_return)
        print("Expected Portfolio Risk (Standard Deviation):", opt_std)
        return None
