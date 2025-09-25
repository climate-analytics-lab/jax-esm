
import pandas as pd



class SystemWatch:

    time : pd.Timestamp
    
    def __init__(
        time : str | pd.Timestamp,
    ):

        self.time = pd.Timestamp(time)

    def advance(
        interval : pd.Timedelta,
    ):            
        self.time += interval


    