to run a GARCHModel
from model import GarchModel

# Instantiate a `GarchModel`
gm_ambuja = GarchModel(ticker="AMBUJACEM.BSE", repo=repo, use_new_data=False)

# Does `gm_ambuja` have the correct attributes?
assert gm_ambuja.ticker == "AMBUJACEM.BSE"
assert gm_ambuja.repo == repo
assert not gm_ambuja.use_new_data
assert gm_ambuja.model_directory == settings.model_directory
