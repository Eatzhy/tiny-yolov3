# Base Image
FROM python:3.6

# create and set working directory
RUN mkdir /app
WORKDIR /app

# Add current directory code to working directory
#ADD . /app/
COPY ./Pipfile /app/
#COPY ./whl/scipy-1.4.1-cp36-cp36m-win_amd64.whl /app/

# set default environment variables
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# set project environment variables
# grab these via Python's os.environ
# these are 100% optional here
#ENV PORT=8080
ENV PORT=30001

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        tzdata \
        python3-setuptools \
        python3-pip \
        python3-dev \
        python3-venv \
        git \
		python3-opencv \
		ffmpeg \
		libsm6 \
		libxext6 \
		wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# install environment dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pipenv

# Install project dependencies
RUN pipenv install --skip-lock --system --dev

#EXPOSE 30001
#CMD gunicorn defectinspection.wsgi:application --bind 0.0.0.0:$PORT