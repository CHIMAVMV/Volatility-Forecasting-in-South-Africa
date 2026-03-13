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
