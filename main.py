import sqlite3

from config import settings
from data import SQLRepository
from fastapi import FastAPI
from model import GarchModel
from pydantic import BaseModel


# Task 8.4.14, `FitIn` class



# Task 8.4.14, `FitOut` class



# Task 8.4.18, `PredictIn` class



# Task 8.4.18, `PredictOut` class



# Task 8.4.15
def build_model():
    # Create DB connection
    connection = ...

    # Create `SQLRepository`
    repo = ...

    # Create model
    model = ...

    # Return model
    return ...


# Task 8.4.9
app = ...


# Task 8.4.11
# `"/hello" path with 200 status code

def hello():
    """Return dictionary with greeting message."""
    return ...


# Task 8.4.16, `"/fit" path, 200 status code

def fit_model():
    """Fit model, return confirmation message.

    Parameters
    ----------
    request : FitIn

    Returns
    ------
    dict
        Must conform to `FitOut` class
    """
    # Create `response` dictionary from `request`


    # Create try block to handle exceptions

        # Build model with `build_model` function
        

        # Wrangle data
       

        # Fit model
        

        # Save model
        

        # Add `"success"` key to `response`
       

        # Add `"message"` key to `response` with `filename`
        

    # Create except block

        # Add `"success"` key to `response`
      

        # Add `"message"` key to `response` with error message
       

    # Return response

    return ...


# Task 8.4.19 `"/predict" path, 200 status code

def get_prediction():
    # Create `response` dictionary from `request`


    # Create try block to handle exceptions

        # Build model with `build_model` function
      

        # Load stored model
     

        # Generate prediction


        # Add `"success"` key to `response`
       

        # Add `"forecast"` key to `response`
       

        # Add `"message"` key to `response`
      

    # Create except block
    
        # Add `"success"` key to `response`
        

        # Add `"forecast"` key to `response`


        #  Add `"message"` key to `response`
  

    # Return response
    return ...
