import numpy as np

# calculations for portfolio features

class Calculations:
    def __init__(self):
        pass

    def current_portfolio_featureset(self, information_list):
        short_correlations, medium_correlations, long_correlations, short_volatilities, medium_volatilities, long_volatilities, \
            short_returns, medium_returns, long_returns, trading_day, portfolio = information_list
        current_port_ave_short_correlations = np.mean(short_correlations[trading_day][portfolio])
        current_port_ave_medium_correlations = np.mean(medium_correlations[trading_day][portfolio])
        current_port_ave_long_correlations = np.mean(long_correlations[trading_day][portfolio])
        current_port_std_short_correlations = np.std(short_correlations[trading_day][portfolio])
        current_port_std_medium_correlations = np.std(medium_correlations[trading_day][portfolio])
        current_port_std_long_correlations = np.std(long_correlations[trading_day][portfolio])
        current_port_ave_short_volatilities = np.mean(short_volatilities[trading_day][portfolio])
        current_port_ave_medium_volatilities = np.mean(medium_volatilities[trading_day][portfolio])
        current_port_ave_long_volatilities = np.mean(long_volatilities[trading_day][portfolio])
        current_port_ave_short_returns = np.mean(short_returns[trading_day][portfolio])
        current_port_ave_medium_returns = np.mean(medium_returns[trading_day][portfolio])
        current_port_ave_long_returns = np.mean(long_returns[trading_day][portfolio])
        current_port_features = np.array([current_port_ave_short_correlations, current_port_ave_medium_correlations, current_port_ave_long_correlations, current_port_std_short_correlations,
                            current_port_std_medium_correlations, current_port_std_long_correlations, current_port_ave_short_volatilities, current_port_ave_medium_volatilities,
                            current_port_ave_long_volatilities, current_port_ave_short_returns, current_port_ave_medium_returns, current_port_ave_long_returns])
        return current_port_features
    
    def potential_port_featureset(self, information_list):
        short_correlations, medium_correlations, long_correlations, short_volatilities, medium_volatilities, long_volatilities, \
            short_returns, medium_returns, long_returns, trading_day, potential_portfolio_configurations = information_list
        possible_port_ave_short_correlations = np.mean(short_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_medium_correlations = np.mean(medium_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_long_correlations = np.mean(long_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_std_short_correlations = np.std(short_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_std_medium_correlations = np.std(medium_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_std_long_correlations = np.std(long_correlations[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_short_volatilities = np.mean(short_volatilities[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_medium_volatilities = np.mean(medium_volatilities[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_long_volatilities = np.mean(long_volatilities[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_short_returns = np.mean(short_returns[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_medium_returns = np.mean(medium_returns[trading_day][potential_portfolio_configurations], axis=1)
        possible_port_ave_long_returns = np.mean(long_returns[trading_day][potential_portfolio_configurations], axis=1)
        potential_port_features = np.array([possible_port_ave_short_correlations, possible_port_ave_medium_correlations, possible_port_ave_long_correlations, possible_port_std_short_correlations,
                    possible_port_std_medium_correlations, possible_port_std_long_correlations, possible_port_ave_short_volatilities, possible_port_ave_medium_volatilities,
                    possible_port_ave_long_volatilities, possible_port_ave_short_returns, possible_port_ave_medium_returns, possible_port_ave_long_returns])
        return potential_port_features
