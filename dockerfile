FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 5000

CMD [ "python","./app.py"]

