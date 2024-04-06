FROM python:3.11.5
WORKDIR /fyp_backend_ml_app
COPY ./requirements.txt /fyp_backend_ml_app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /fyp_backend_ml_app/requirements.txt
COPY ./ /fyp_backend_ml_app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]