FROM  pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND=nointeractive
RUN pip install --upgrade pip
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN apt-get update
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]