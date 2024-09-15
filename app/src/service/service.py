# service.py
# ==================================================
# standard
from typing import Any, Collection, Tuple, List, Dict
import os
# requirements
import yfinance as yf
import numpy as np
import pandas
# defined
from constants.constants import RISK_FREE_RATE, TRADING_DAYS_YEAR
from utils.optimization import Optimization
# --------------------------------------------------

class Analysis(object):
    
    @staticmethod
    def yf_download_historical(tickers: List[str], start_date: str, end_date: str) -> pandas.DataFrame:
        '''Download historical data from Yahoo Finance'''
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data
    
    @staticmethod
    def read_investing_csv(dir_path: str) -> pandas.DataFrame:
        '''Raw data from investing.com to Pandas Dataframe'''
        rd_dir = dir_path.strip('/')
        ap_func = lambda rd_dir, f: {'path': f'{rd_dir}/{f}', 'code': f.split('.')[0]}
        investing_files = [ap_func(rd_dir, f) for f in os.listdir(rd_dir)]
        
        data = pandas.DataFrame()
        for f in investing_files:
            _df = pandas.read_csv(f['path'], usecols=['Date', 'Price'], thousands=',')
            _df = _df.rename(columns={'Price': f['code']})
            _df['Date'] = pandas.to_datetime(_df['Date'], format='%m/%d/%Y')
            _df = _df.set_index('Date').sort_index()
            data = pandas.concat([data, _df], axis=1)
        return data.dropna()
    
    @staticmethod
    def read_nasdaq_csv(dir_path: str) -> pandas.DataFrame:
        '''Raw data from nasdaq.com to Pandas Dataframe'''
        rd_dir = dir_path.strip('/')
        ap_func = lambda rd_dir, f: {'path': f'{rd_dir}/{f}', 'code': f.split('.')[0]}
        investing_files = [ap_func(rd_dir, f) for f in os.listdir(rd_dir)]
        
        data = pandas.DataFrame()
        for f in investing_files:
            _df = pandas.read_csv(f['path'], usecols=['Date', 'Close/Last'])
            _df = _df.rename(columns={'Close/Last': f['code']})
            _df['Date'] = pandas.to_datetime(_df['Date'], format='%m/%d/%Y')
            _df = _df.set_index('Date').sort_index()
            data = pandas.concat([data, _df], axis=1)
        return data.dropna()
    
    @staticmethod
    def daily_log_returns(data: pandas.DataFrame) -> Any:
        '''Log daily returns as ln(Pt/Pt-1).'''
        return np.log(data).diff().dropna()
    
    @staticmethod
    def estimate_assets_beta(returns: pandas.DataFrame, benchmark_returns: pandas.Series) -> Dict[str, float]:
        assets_beta = {}
        for a in returns.columns:
            covariance_matrix = np.cov(returns[a], benchmark_returns)
            assets_beta[a] = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        return assets_beta
    
    @staticmethod
    def annualized_sharpe_ratio(returns: Any) -> Collection:
        mean_returns = np.exp(returns.mean() * TRADING_DAYS_YEAR) - 1
        volatility = np.sqrt(returns.var(ddof=0) * TRADING_DAYS_YEAR)
        sharpe_ratios = ((mean_returns - RISK_FREE_RATE) / volatility).sort_values(ascending=0)
        sharpe_ratios.name = 'sharpe_ratio'
        return sharpe_ratios
    
    @staticmethod
    def annualized_sortino_ratio(returns: Any) -> Collection:
        mean_returns = np.exp(returns.mean() * TRADING_DAYS_YEAR) - 1
        neg_volatility = np.sqrt(np.minimum(returns, 0).var(ddof=0) * TRADING_DAYS_YEAR)
        sortino_ratios = ((mean_returns - RISK_FREE_RATE) / neg_volatility).sort_values(ascending=0)
        sortino_ratios.name = 'sortino_ratio'
        return sortino_ratios
    
    @staticmethod
    def estimate_jensen_alpha(assets_beta: Any, returns: pandas.DataFrame, benchmark_returns: pandas.Series):
        market_excess_return = (np.exp(np.mean(benchmark_returns) * TRADING_DAYS_YEAR) - 1) - RISK_FREE_RATE
        mean_returns = np.exp(returns.mean() * TRADING_DAYS_YEAR) - 1
        jensen_alpha = mean_returns - (RISK_FREE_RATE + np.array([v for v in assets_beta.values()]) * market_excess_return)
        jensen_alpha.name = 'jensen_alpha'
        return jensen_alpha
    
    @staticmethod
    def top_sorted_sortino(sortino_ratios: Collection) -> Any:
        '''Sort by Sortino ratio (from highest to lowest)'''
        sorted_sortino = sortino_ratios.sort_values(ascending=False)
        return sorted_sortino.head(5)
    
    @staticmethod
    def top_tickers_returns(data: pandas.DataFrame, top_tickers: Any) -> Collection:
        '''Top tickers data'''
        top_data = data[top_tickers]
        return np.log(top_data).diff().dropna()
    
    @staticmethod
    def top_tickers_stats(top_returns_data: Any) -> Tuple[Any]:
        '''Calculate mean returns and covariance matrix'''
        top_mean_returns = np.exp(top_returns_data.mean() * TRADING_DAYS_YEAR) - 1
        top_volatility = np.sqrt(top_returns_data.var(ddof=0) * TRADING_DAYS_YEAR)
        top_cov_matrix = top_returns_data.cov() * 252
        return top_mean_returns, top_volatility, top_cov_matrix
    
    @staticmethod
    def optimize_portfolio(top_mean_returns: Any, top_cov_matrix: Any) -> Any:
        '''Optimize portfolio'''
        optimal_result = Optimization.optimize_portfolio(top_mean_returns, top_cov_matrix, RISK_FREE_RATE)
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
        print('Assets:', ' '.join(top_mean_returns.index))
        print('Optimal Portfolio Weights:', optimal_weights)
        print('Expected Portfolio Return:', opt_return)
        print('Expected Portfolio Risk (Standard Deviation):', opt_std)
        return None
