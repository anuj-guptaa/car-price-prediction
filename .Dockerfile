FROM python:3.11.4-bookworm

RUN apt-get update

ADD requirements.txt /code/

RUN pip3 install -r code/requirements.txt

CMD tail -f /dev/null
