#start with a lightweight python image
# gives our container a python runtime
FROM python:3.14-slim

# Set the working directory in side the conatiner
WORKDIR /app

#Copy the dpeendencies list before copying the application code
COPY requirements.txt .

#avoid retaining pip's download cache in the image
RUN pip install --no-cache-dir -r requirements.txt

#Copy the application code and save the model into /app
COPY . .

#DOcument the port used by FastAPI application
EXPOSE 8000

#start the API server when the conatier launches
#0.0.0.0 allows connections through the conatiners network interface
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]