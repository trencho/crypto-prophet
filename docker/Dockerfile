FROM python:latest

LABEL maintainer='Aleksandar Trenchevski <atrenchevski@gmail.com>'

ARG DEBIAN_FRONTEND=noninteractive

ENV USER=app
ENV APP_HOME=/home/app/web
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN pip3 install --upgrade pip
COPY ./requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

RUN adduser --system --group $USER
RUN mkdir $APP_HOME
WORKDIR $APP_HOME

COPY . /
RUN chown -R $USER:$USER $APP_HOME
USER $USER

EXPOSE 5000