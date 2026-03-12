import os
from glob import glob

import joblib
import pandas as pd
from arch import arch_model
from config import settings
from data import AlphaVantageAPI, SQLRepository


class GarchModel:
    """Class for training GARCH model and generating predictions.

    Atttributes
    -----------
    ticker : str
        Ticker symbol of the equity whose volatility will be predicted.
    repo : SQLRepository
        The repository where the training data will be stored.
    use_new_data : bool
        Whether to download new data from the AlphaVantage API to train
        the model or to use the existing data stored in the repository.
    model_directory : str
        Path for directory where trained models will be stored.

    Methods
    -------
    wrangle_data
        Generate equity returns from data in database.
    fit
        Fit model to training data.
    predict
        Generate volatilty forecast from trained model.
    dump
        Save trained model to file.
    load
        Load trained model from file.
    """

    def __init__():
        self.ticker = ... 
        self.repo = ...
        self.use_new_data = ...
        self.model_directory = ...

    def wrangle_data():
        """Extract data from database (or get from AlphaVantage), transform it
        for training model, and attach it to `self.data`.

        Parameters
        ----------
        n_observations : int
            Number of observations to retrieve from database

        Returns
        -------
        None
        """
        # Add new data to database if required
        

        # Pull data from SQL database
        df = ...

        # Clean data, attach to class as `data` attribute


        self.data = ...

    def fit():
        """Create model, fit to `self.data`, and attach to `self.model` attribute.
        For assignment, also assigns adds metrics to `self.aic` and `self.bic`.

        Parameters
        ----------
        p : int
            Lag order of the symmetric innovation

        q : ind
            Lag order of lagged volatility

        Returns
        -------
        None
        """
        # Train Model, attach to `self.model`
        self.model = ...
        

    def __clean_prediction():
    
        """Reformat model prediction to JSON.

        Parameters
        ----------
        prediction : pd.DataFrame
            Variance from a `ARCHModelForecast`

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Calculate forecast start date
        

        # Create date range
        

        # Create prediction index labels, ISO 8601 format
        

        # Extract predictions from DataFrame, get square root
        

        # Combine `data` and `prediction_index` into Series
        

        # Return Series as dictionary
        
        return ...

    def predict_volatility():
        """Predict volatility using `self.model`

        Parameters
        ----------
        horizon : int
            Horizon of forecast, by default 5.

        Returns
        -------
        dict
            Forecast of volatility. Each key is date in ISO 8601 format.
            Each value is predicted volatility.
        """
        # Generate variance forecast from `self.model`
        prediction = ...

        # Format prediction with `self.__clean_predction`
        prediction_formatted = ...

        # Return `prediction_formatted
        return ...

    def dump():
        """Save model to `self.model_directory` with timestamp.

        Returns
        -------
        str
            filepath where model was saved.
        """
        # Create timestamp in ISO format
        timestamp = ...
        # Create filepath, including `self.model_directory`
        filepath = ...
        # Save `self.model`
        
        # Return filepath
        return ...

    def load():
        """Load most recent model in `self.model_directory` for `self.ticker`,
        attach to `self.model` attribute.

        """
        # Create pattern for glob search
        pattern = ...
        # Use glob to get most recent model, handle errors
        
        # Load model and attach to `self.model`
        self.model = ...
