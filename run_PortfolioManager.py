import training_trading as t_train
import test_trading as t_test

print('\nTraining')
train_portfolio_final, train_market_final, train_portfolio_vol, train_market_vol = t_train.PortfolioManagerTraining()
print('\nTesting')
test_portfolio_final, test_market_final, test_portfolio_vol, test_market_vol = t_test.PortfolioManagerTest()

print(f'\nTrain Period Portfolio End Value (Base 100): {train_portfolio_final:.2f}, Train Period Market End Value (Base 100): {train_market_final:.2f}')
print(f'Train Period Portfolio Volatility: {train_portfolio_vol:.2f}, Train Period Market Volatility: {train_market_vol:.2f}\n')

print(f'Test Period Portfolio End Value (Base 100): {test_portfolio_final:.2f}, Test Period Market End Value (Base 100): {test_market_final:.2f}')
print(f'Test Period Portfolio Volatility: {test_portfolio_vol:.2f}, Test Period Market Volatility: {test_market_vol:.2f}\n')
