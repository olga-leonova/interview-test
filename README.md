
**Prerequisits** make sure you have installed docker locally https://www.docker.com/

To start work run the command:
`make init`

It will prepare a virtual environment (pipenv) and install all required packages.

**Part 1: SQL**

- To prepare docker container with MySQL database run the command:
`make docker`
 
    This step is needed to prepare and start the docker container with MySQL DB inside. The MySQL database has `test_db.test_table`
 table, which is created from testset_B.csv

- To extract info from DB, calculate and save features use `run_features_to_csv.py`
This script requires the output folder path as an argument.

**Part 2: ML**

- To train ML model use `run_train_model.py` This scrip requires data file path as an argument.
It will train tfidf and lighgbm model and save them into `model` folder (default ml models path). 

- To run REST api and generate predictions for single article text execute:
1. `pipenv shell`
2. `python api.py`
3. In separate terminal execute:

    `curl -X GET -H "Content-Type: application/json" -d '{"text_article": "<line_wo_<>>"}' http://localhost:5000/productid`

    Example:
    
    `curl -X GET -H "Content-Type: application/json" -d '{"text_article": "waschvollautomat kg kapazitaet"}' http://localhost:5000/productid`


If you are interested in ML preparation check jupyter notebook `/jupyter_notebook/interview_ml_prep.ipynb`

**Further steps:**
1. Add tests
2. Data cleaning
3. Investigate usage of  ‘manufacturer’ field as single feature
4. Add accuracy check using last available model vs newly generated
5. Add logging
6. Check non-default pathes (model and data)
7. Add cross validation# interview-test
