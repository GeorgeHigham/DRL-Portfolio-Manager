# DRL-Portfolio-Manager
Building a Deep Reinforcement Learner to manage a stock portfolio

This project uses Deep Reinforcment Learning to manage a portfolio over time, trading stocks based on features created using historical prices. Limitations were set on the portfolio to reduce considerations for the project. The limitations include: the portfolio manager is only able to make 1 trade a day, no transaction costs to trades, fixed portfolio size, portfolio can only own 1 share in each stock. 

The portfolio manager outperformed a 'market' consisting of 1 share each of the available stock. However, as there were no transaction costs in trading this may not be the case in application.

<img src="visualisations/Training Graph.png" alt="alt text" width="600" height="450">
<img src="visualisations/Test Graph.png" alt="alt text" width="600" height="450">

# References and further notes
Stocks data from: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data

The csv data has been separated to be uploaded to GitHub, data_processing is used on a single folder that includes every csv in the zip files. However, training and testing can be run using only the train and test arrays as this is what is produced from the data_processing.py file.
