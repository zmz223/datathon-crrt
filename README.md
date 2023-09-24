# datathon-crrt

### How to run ###
* Create a ```.env``` file with the credentials to access the AWS account with the permissions to access the data. Example:<br>
<code>
AWS_ACCESS_KEY_ID=xxxxxxxx<br>
AWS_SECRET_ACCESS_KEY=yyyyyyyy
</code>
* Run the ```get_data.py``` script to query the data from the AWS bucket. The script accepts 2 arguments: ```--filename``` which is the name of the saved file, and ```--aki_filename``` which is the path to the AKI score csv file (by default it is set to ```max_aki_score.csv```). To customize you run modify the ```config/get_data.py``` file;
* Run the ```prediction.py``` script. This script runs an analysis on the selected dataset and creates a results sub-folder with all the analysis results. To customize you run modify the ```config/predictions.py``` file.

### Dependencies ###
Check the ```requirements.txt``` file.