FROM python:3.14.7-slim

LABEL maintainer="Aleksandar Trenchevski <atrenchevski@gmail.com>"

ARG DEBIAN_FRONTEND=noninteractive

ENV USER=app
ENV APP_HOME=/home/app/web

# Prevents Python from writing pyc files and keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies.
#
# `apt-get`, not `apt`: apt's CLI is explicitly not stable for scripting and says so on every run.
#
# No `upgrade`/`dist-upgrade`. They defeat the pinned base image -- the same tag would produce a
# different package tree on every build, which is exactly what pinning exists to prevent, and it
# makes a failure impossible to reproduce from the tag alone. System-package currency comes from
# moving the pinned base image instead, which Dependabot does in a reviewable PR.
#
# nginx dropped: installed and referenced by nothing. No file in this repo configures or starts it
# -- the app is served by gunicorn behind traefik, per docker-compose.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 tzdata && \
    apt-get clean && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory
WORKDIR $APP_HOME

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container.
COPY . .

RUN adduser --system --group $USER && \
    chown -R $USER:$USER $APP_HOME
USER $USER

EXPOSE 5000