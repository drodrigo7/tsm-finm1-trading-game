from typing import List, Any, Tuple, Collection
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from constants.constants import MINIMUM_WEIGHT, MAXIMUM_WEIGHT

class Optimization(object):
    
    @staticmethod
    def portfolio_performance(weights, mean_returns: float, cov_matrix) -> Tuple[Collection, float]:
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, std
    
    @staticmethod
    def negative_sharpe(weights, mean_returns: float, cov_matrix, risk_free_rate: float) -> float:
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio
    
    @classmethod
    def optimize_portfolio(cls, mean_returns: float, cov_matrix, tickers: List[str], risk_free_rate: float) -> Any:
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix, risk_free_rate)
        constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        bounds = tuple((MINIMUM_WEIGHT, MAXIMUM_WEIGHT) for _ in range(num_assets))
        initial_weights = np.array([1/len(tickers)] * len(tickers))
        
        result = minimize(
            cls.negative_sharpe, initial_weights, args=args,
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        return result
    
    @classmethod
    def generate_portfolios(cls, num_portfolios: int, mean_returns: float, cov_matrix, risk_free_rate: float) -> Tuple[Collection, Collection]:
        results = np.zeros((3, num_portfolios))
        weights_record = []
        
        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return, portfolio_stddev = cls.portfolio_performance(weights, mean_returns, cov_matrix)
            results[0,i] = portfolio_return
            results[1,i] = portfolio_stddev
            results[2,i] = (portfolio_return - risk_free_rate) / portfolio_stddev
        
        return results, weights_record
    
    @staticmethod
    def efficient_frontier_plot(results: Collection) -> None:
        plt.figure(figsize=(10, 7))
        plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o')
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Risk (Standard Deviation)')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        plt.show()
