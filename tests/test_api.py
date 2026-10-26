# Imprt test client so we can send simulated HTTP requests
from fastapi.testclient import TestClient

# Import the application object defined in main.py
from main import app

# Create a reusable test client for all test functions.
client = TestClient(app)

def test_health_endpoint():
    # """
    # Verify that the application is running and can respond to a health-check request
    # """
    response = client.get("/health")
    # A successful HTTP request should return status code 200
    assert response.status_code == 200

    # Now check response body
    assert response.json() == {"status":"healthy"}

def test_valid_iris_predictipon():
    # """
    # Verify that valid flower measurements produce a prediction
    # """
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length":1.6,
        "petal_width": 0.5
    }
    
    # Send a simulated HTTP post request with json input
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    #Convert the json response into a python dictionary
    result = response.json()

    #Now check the response contract: both fields must exists
    assert "prediction" in result
    assert "class_name" in result

    #The predicted class must be one of three Iris classes
    assert result["class_name"] in ["setosa","versicolor","virginica"]
    assert result["prediction"] in [0,1,2]

    def test_invalid_iris_input():
        """
        Verify that the inputs are rejected by FastAPI
        """
        payload = {
            "sepal_length": -5.1,
            "sepal_width": 3.5,
            "petal_length":1.6,
            "petal_width": 0.5
        }
        response = client.post("/predict", json = payload)
        assert response.status_code == 200