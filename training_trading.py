import numpy as np
import random
from learner import DQNAgent
import torch
from calculations import Calculations
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime

def PortfolioManagerTraining():

    # load stock prices, dates, and features
    stocks_array = np.load('train_arrays/train_stocks_array.npy')
    dates = np.load('train_arrays/train_dates.npy', allow_pickle=True)[90:-1]
    dates = [datetime.strptime(date, '%d/%m/%y') for date in dates]
    short_correlations = np.load('train_arrays/train_short_correlations.npy')
    medium_correlations = np.load('train_arrays/train_medium_correlations.npy')
    long_correlations = np.load('train_arrays/train_long_correlations.npy')
    short_volatilities = np.load('train_arrays/train_short_volatilities.npy')
    medium_volatilities = np.load('train_arrays/train_medium_volatilities.npy')
    long_volatilities = np.load('train_arrays/train_long_volatilities.npy')
    short_returns = np.load('train_arrays/train_short_returns.npy')
    medium_returns = np.load('train_arrays/train_medium_returns.npy')
    long_returns = np.load('train_arrays/train_long_returns.npy')

    calcs = Calculations()
    random.seed(0)
    np.random.seed(0)
    replay_start = 200
    longer_period = 90
    trading_end = stocks_array.shape[1]-1
    agent = DQNAgent(0)

    batch_size = 50
    portfolio_size = 15

    # initial random portfolio
    portfolio = np.array(np.random.randint(0, 259, portfolio_size))

    portfolio_prices = np.array(stocks_array[:, 89][portfolio]) # first day before trading begins
    market_prices = np.array(stocks_array[:, 89]) # first day before trading begins

    portfolio_value = 100
    market_value = 100

    portfolio_value_tracker = []
    market_value_tracker = []

    for trading_day in range(longer_period, trading_end):
        
        current_port_features = calcs.current_portfolio_featureset((short_correlations, medium_correlations, long_correlations, short_volatilities, medium_volatilities, long_volatilities,
                short_returns, medium_returns, long_returns, trading_day, portfolio))

        # get current trading day prices
        portfolio_today_prices = np.array(stocks_array[:, trading_day][portfolio])
        market_today_prices = np.array(stocks_array[:, trading_day])
        # return for portfolio and market calculated using the portfolio periods prices saved in last loop
        port_period_return = (portfolio_today_prices - portfolio_prices)/portfolio_prices
        market_return = (market_today_prices - market_prices) / market_prices
        # keep track of portfolio and market value over time
        portfolio_value *= 1+np.mean(port_period_return)
        portfolio_value_tracker.append(portfolio_value)
        market_value *= 1+np.mean(market_return)
        market_value_tracker.append(market_value)

        # now go to change portfolio for next period
        # get all potential portfolios with 1 trade made
        incomplete_portfolios = np.array([np.delete(portfolio, i) for i in range(len(portfolio))])
        potential_portfolio_configurations = []
        for incomplete in incomplete_portfolios:
            complete_portfolio = [np.insert(incomplete,0,num) for num in range(259) if num not in portfolio]
            potential_portfolio_configurations.extend(complete_portfolio)
        potential_portfolio_configurations = np.array(potential_portfolio_configurations)
        # get features for all possible portfolio configurations
        potential_port_features = np.array(calcs.potential_port_featureset((short_correlations, medium_correlations, long_correlations, short_volatilities, medium_volatilities, long_volatilities,
                short_returns, medium_returns, long_returns, trading_day, potential_portfolio_configurations))).T
        
        # passing all possible next states through as a proxy for passing through state and actions
        # returns index of selected action from learner or random
        action_index = agent.act(potential_port_features)
        # choosing next portfolio using index and getting its features
        portfolio = potential_portfolio_configurations[action_index]
        chosen_port_features = potential_port_features[action_index]
        # update portfolio and market prices so return for next loop is based off updated portfolio
        portfolio_prices = np.array(stocks_array[:, trading_day][portfolio])
        market_prices = np.array(stocks_array[:, trading_day])

        # creating a target variable for the learner
        next_day_return = np.mean(stocks_array[:, trading_day+1][portfolio] - stocks_array[:, trading_day][portfolio] / stocks_array[:, trading_day][portfolio])

        agent.remember(current_port_features, next_day_return, chosen_port_features)

        if trading_day % 50 == 0:
            print(f"Trading Day: {trading_day}/{trading_end}, P&L: {portfolio_value:.5f}, Portfolio: {portfolio}") 

        if trading_day > replay_start:
            agent.replay(batch_size)

    torch.save(agent.model, f'Portfolio Manager')

    portfolio_final = 100 * portfolio_value_tracker[-1] / portfolio_value_tracker[0]
    market_final = 100 * market_value_tracker[-1] / market_value_tracker[0]
    portfolio_vol = np.std(portfolio_value_tracker)
    market_vol = np.std(market_value_tracker)

    plt.plot(dates, market_value_tracker, label='market tracker', color='tan')
    plt.plot(dates, portfolio_value_tracker, label='portfolio tracker', color='seagreen')

    plt.xlabel('Trading Days (MM/YY)')
    plt.ylabel('Return % (Base 100)')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
    plt.grid()
    plt.legend()
    plt.title('DRL Portfolio Manager Training Performance')
    plt.savefig('visualisations/Training Graph', dpi=300)
    plt.clf()

    return portfolio_final, market_final, portfolio_vol, market_vol


