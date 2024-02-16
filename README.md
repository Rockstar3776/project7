# project7
All code related to OpenClassrooms project 7

## Prerequisites
To install Python project dependencies run following command.
`pip install -r requirements.txt` 

## API

### Start the application

To start the project API, run following command.
`uvicorn api.main:app --reload`

### Docker

1. Build the image: `docker build -t api .`
2. Run the image: `docker run -p 80:80 api`

### About FastAPI
https://fastapi.tiangolo.com/#example

## Dashboard

### Start the application

To start the project dashboard, run following command
`streamlit run dashboard/main.py`

### About Streamlit
https://docs.streamlit.io/library/api-reference


## About Azure

Login to Azure ECR: `docker login projet7fastapi.azurecr.io -u projet7fastapi -p XXX`

Build the image: `docker build -t projet7fastapi.azurecr.io/api:latest .`

Push the image: `docker push projet7fastapi.azurecr.io/api:latest`