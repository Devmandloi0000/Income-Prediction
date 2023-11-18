# Adult Census Income-Prediction-
## End To End Project 


This is a classification problem where we need to predict whether a person earns more than a sum of 50,000 k anuually or not. This classification task is accomplished by using a SVC Classifier trained on the dataset extracted by Barry Becker from the 1994 Census database. The dataset contains about 33k records and 15 features which after all the implementation of all standard techniques like Data Cleaning, Feature Engineering, Feature Selection, Outlier Treatment, etc was feeded to our Classifier which after training and testing, was deployed in the form of a web application.

## Project Goal

This end-to-end project is made as a part of data science internship for Ineuron.ai.

## Files Aspects or Code Flow details :-

The whole code project pipeline is divided into following ways

> notebook                              :- Folder for ipynb file and location of raw data

    > data                              :- Raw data location

    > EDA                               :- For analysis

        > EDA.ipynb                     :- For expolatory data analysis 

        > final_model.ipynb             :-

        > model_training.ipynb          :- Model building and model evaluation

> src

    > exception.py                      :- Handling custom exception error 

    > logger.py                         :- For logging

    > utils.py                          :- Handling all the neccessary file or we can say genric file

    > components                        :- folder for preprocessing and model training

        > data_filestation.py           :- Handling the artifacts folder file type and used to stored location of all the neccessary file

        > data_ingestion.py             :- Used for data spliting and return a train and test split data in csv format

        > data_transfomation.py         :- Mainly used for data cleaning preprocessing and creating preprocessed pickle file

        > model_training.py             :- Model trained on behalf of pickle file and enhenced the 

    > pipeline

        > data_storage.py               :- Used to connect with database and creating connection 

        > prediction_pipeline.py        :- Used for converting list of new data into dataframe

        > training_pipeline.py          :_ Used for flow of all the files

    > app.py                            :- By using Flask to create a web app

    > requirements.txt                  :- Installing all the lib at one short

    > templates                         :- For creating user interface 

    > static                            :- For adding style to web page

    > setup.py                          :- For setup whole folder as a module 

![Alt text](<data flow.png>)


## Technical Aspects 

The whole project has been divided into three parts. These are listed as follows :

    • Data Preparation : This consists of our data,With this Data we are performing Data Cleaning, Feature Engineering, Feature Selection, train and test split of data and provide data as for preprocess in form of pickle file which been used for train validation.

    • Model Training   : In this model is trained here we already did perform the model on top of the best 

    • Data Base        : MongoDB atls is used for storing the new test data  

    • Model Development : In this step, we use the resultant data after the implementation of the previous step to cross validate our Machine Learning model and perform Hyperparameter optimization based on various performance metrics in order to make our model predict as accurate results as possible.

    • Model Deployment : This step include creation of a front-end using Amazon AWS and Elastic beanstalk.


## Installation 

In this code Python==3.8 is used for project if you want to install in your system then you can visit https://www.python.org/downloads/ this link ,  To install the required packages and libraries, run this command in the project directory after cloning the repository

>pip install -r requirements.txt


## Technology used 

If you want to read you can visit this link 

> Python        :- https://www.python.org/downloads/

> Pandas lib    :- https://pandas.pydata.org/pandas-docs/version/0.15/tutorials.html

> Numpy         :- https://numpy.org/doc/stable/user/absolute_beginners.html

> Seaborn       :- https://seaborn.pydata.org/tutorial.html

> Matplotlib    :- https://matplotlib.org/stable/index.html

> Flask         :- https://flask.palletsprojects.com/en/3.0.x/

> Scikit-learn  :- https://scikit-learn.org/stable/modules/classes.html

> Scipy         :- https://docs.scipy.org/doc/scipy/

> Pymongo       :- https://pymongo.readthedocs.io/en/stable/

> MLflow        :- https://mlflow.org/docs/0.2.0/index.html

> Amazon AWS    :- https://aws.amazon.com/console/


## Appendix

> Link for App Documentation :- https://github.com/Devmandloi0000/Income-Prediction

> Link for Youtube video for description of the project :- 

> Deployment link of the app :- https://cs7mymkatp.us-east-1.awsapprunner.com 




