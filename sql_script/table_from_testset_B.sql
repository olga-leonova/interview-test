CREATE DATABASE test_db;

USE test_db;

CREATE TABLE IF NOT EXISTS test_table 
(productid INT, 
brand VARCHAR(20), 
ram_gb TINYINT,
hdd_gb INT,
ghz FLOAT,
price INT);

LOAD DATA INFILE '/opt/MLE_Task/testset_B.tsv' 
INTO TABLE test_table 
FIELDS TERMINATED BY '\t' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;
