to run a GARCHModel
from model import GarchModel

# Instantiate a `GarchModel`
gm_ambuja = GarchModel(ticker="AMBUJACEM.BSE", repo=repo, use_new_data=False)

# Does `gm_ambuja` have the correct attributes?
assert gm_ambuja.ticker == "AMBUJACEM.BSE"
assert gm_ambuja.repo == repo
assert not gm_ambuja.use_new_data
assert gm_ambuja.model_directory == settings.model_directory






# Instantiate `GarchModel`, use new data
model_shop = GarchModel(ticker="SHOPERSTOP.BSE", repo=repo, use_new_data=True)

# Check that model doesn't have `data` attribute yet
assert not hasattr(model_shop, "data")

# Wrangle data
model_shop.wrangle_data(n_observations=1000)

# Does model now have `data` attribute?
assert hasattr(model_shop, "data")

# Is the `data` a Series?
assert isinstance(model_shop.data, pd.Series)

# Is Series correct shape?
assert model_shop.data.shape == (1000,)

model_shop.data.head()










# Instantiate `GarchModel`, use old data
model_shop = GarchModel(ticker="SHOPERSTOP.BSE", repo=repo, use_new_data=False)

# Wrangle data
model_shop.wrangle_data(n_observations=1000)

# Fit GARCH(1,1) model to data
model_shop.fit(p=1, q=1)

# Does `model_shop` have a `model` attribute now?
assert hasattr(model_shop, "model")

# Is model correct data type?
assert isinstance(model_shop.model, ARCHModelResult)

# Does model have correct parameters?
assert model_shop.model.params.index.tolist() == ["mu", "omega", "alpha[1]", "beta[1]"]

# Check model parameters
model_shop.model.summary()





# Generate prediction from `model_shop`
prediction = model_shop.predict_volatility(horizon=5)

# Is prediction a dictionary?
assert isinstance(prediction, dict)

# Are keys correct data type?
assert all(isinstance(k, str) for k in prediction.keys())

# Are values correct data type?
assert all(isinstance(v, float) for v in prediction.values())

prediction





# Save `model_shop` model, assign filename
filename = model_shop.dump()

# Is `filename` a string?
assert isinstance(filename, str)

# Does filename include ticker symbol?
assert model_shop.ticker in filename

# Does file exist?
assert os.path.exists(filename)

filename




# Assign load output to `model`
model_shop = load(ticker="SHOPERSTOP.BSE")

# Does function return an `ARCHModelResult`
assert isinstance(model_shop, ARCHModelResult)

# Check model parameters
model_shop.summary()







from main import build_model

# Instantiate `GarchModel` with function
model_shop = build_model(ticker="SHOPERSTOP.BSE", use_new_data=False)

# Is `SQLRepository` attached to `model_shop`?
assert isinstance(model_shop.repo, SQLRepository)

# Is SQLite database attached to `SQLRepository`
assert isinstance(model_shop.repo.connection, sqlite3.Connection)

# Is `ticker` attribute correct?
assert model_shop.ticker == "SHOPERSTOP.BSE"

# Is `use_new_data` attribute correct?
assert not model_shop.use_new_data

model_shop











model_shop = GarchModel(ticker="SHOPERSTOP.BSE", repo=repo, use_new_data=False)

# Check that new `model_shop_test` doesn't have model attached
assert not hasattr(model_shop, "model")

# Load model
model_shop.load()

# Does `model_shop_test` have model attached?
assert hasattr(model_shop, "model")

model_shop.model.summary()

