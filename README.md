# Volatility-Forecasting-in-South-Africa
Volatility Forecasting in South Africa stock Market
use this to connect to the Database
connection = sqlite3.connect(settings.db_name, check_same_thread=False)
repo = SQLRepository(connection=connection)

print("repo type:", type(repo))
print("repo.connection type:", type(repo.connection))
