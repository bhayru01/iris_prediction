FROM continuumio/anaconda3:4.4.0 
COPY ./ /usr/local/python/
EXPOSE 1993
WORKDIR /usr/local/python/
RUN pip install -r ./app/requirements.txt
CMD python ./app/flask_predict_api.py












