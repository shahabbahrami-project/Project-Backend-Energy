###########
# BUILDER #
###########

# pull official base image
FROM python:3.8-slim AS builder


# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# lint
RUN pip install --upgrade pip
COPY ./app .

# install dependencies
COPY ./requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /usr/src/app/wheels -r requirements.txt





#########
# FINAL #
#########

# pull official base image
FROM python:3.8-slim

# install opnssl
RUN apt-get update
RUN apt-get install libssl-dev -y
RUN apt-get install sudo -y
RUN apt-get install wget -y
# install vim
RUN apt-get install vim -y
RUN apt-get install libatlas-base-dev gfortran -y
RUN apt-get install python-dev -y
RUN apt-get install python-pip -y
RUN apt-get install python-numpy python-scipy -y
RUN pip install cvxpy
RUN apt-get install python-nose -y

# create the app user
#RUN useradd app
RUN useradd -p $(openssl passwd -1 app) app
# add to sudo group
RUN usermod -aG sudo app

# create directory for the app user
RUN mkdir -p /home/app

# create the appropriate directories
ENV HOME=/home/app
ENV APP_HOME=/home/app/web
RUN mkdir $APP_HOME
RUN mkdir $APP_HOME/staticfiles
RUN mkdir $APP_HOME/mediafiles
RUN mkdir $APP_HOME/db-config
RUN mkdir -p /var/log/gunicorn
RUN chmod 777 /var/log/*
WORKDIR $APP_HOME

# install dependencies
COPY --from=builder /usr/src/app/wheels /wheels
COPY --from=builder /usr/src/app/requirements.txt .
RUN pip install --no-cache /wheels/*


# copy project
COPY ./app $APP_HOME

# chown all the files to the app user
RUN chown -R app:app $APP_HOME
RUN chown -R app:app /var/log

# change to the app user
USER app
