import pandas as pd
from mysql.connector import connect
from interview_test.config import *


def sql_to_pandas(qry: str) -> pd.DataFrame:
    """
    create connection to mysql db, fetch info using input query and return dataframe with extracted info
    :param qry: sdting with query to execute
    :return: pandas dataframe with fetched info
    """
    conn = connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME)

    with conn.cursor() as cursor:
        print(qry)
        cursor.execute(qry)
        rows = cursor.fetchall()
        cols = cursor.column_names
    return pd.DataFrame(rows, columns=cols)
