FROM mysql:latest

ENV MYSQL_ROOT_PASSWORD root

#COPY ./MLE_Task/testset_B.tsv /opt/MLE_Task/testset_B.tsv

#COPY ./sql_script/table_from_testset_B.sql /opt/sql_script/table_from_testset_B.sql

COPY ./sql_script/table_from_testset_B.sql /docker-entrypoint-initdb.d/table_from_testset_B.sql

EXPOSE 3306


#COPY Pipfile* /opt/slice-signals-extraction/
#RUN cd /opt/slice-signals-extraction/ && pipenv lock --requirements > requirements.txt

#RUN pip install -r /opt/slice-signals-extraction/requirements.txt

#COPY . /opt/slice-signals-extraction
#WORKDIR /opt/slice-signals-extraction