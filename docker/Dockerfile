FROM python:3.12-slim

LABEL maintainer="Aleksandar Trenchevski <atrenchevski@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

ENV USER=app
ENV APP_HOME=/home/app/web
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt update && \
    apt upgrade -y && \
    apt dist-upgrade -y && \
    apt install -y --no-install-recommends libgomp1 nginx tzdata && \
    apt autoclean clean && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory
WORKDIR $APP_HOME

COPY ./requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN adduser --system --group $USER && \
    chown -R $USER:$USER $APP_HOME
USER $USER

EXPOSE 5000