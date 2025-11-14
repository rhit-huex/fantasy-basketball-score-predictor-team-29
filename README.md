# fantasy-basketball-score-predictor-team-29
Rishi Ramesh, Henry Sheffield, Kelvin Zhou, Evan Hu

### app.py
<b>This is our website that allows users to select any current NBA player and any date from the 2024-25 season to calculate the predicted Fantasy points for the player's next game after the given date.


### baseline_nba.ipynb
<b>This is where we curated our datasets and calculated baselines.</b> The initial data set was able to be run in a single run through of the api. The second larger data set, around ~26,000 -> ~50,000 rows, needed to be split into two different runs of API calls which is what is currently implemented in the `ipynb`. 

### data_processing.py
<b> This is where the data preprocessing is done, and the creation of the sequences</b>. We did try to use opponents and players teams but the accuracy actually went down when we used it for the test set. We think this is likely because of over fitting.

### model.py
<b> This is where we train the model and then save the model for future use.</b> We also define the methods in here to 

### predictions.py
<b> Command line based predictions to test out the various applicaitons that we were able to create</b>


