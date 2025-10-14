import random

from locust import HttpUser, between, task


class MLUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        features = [random.random() for _ in range(5)]
        self.client.post("/predict", json={"features": features})
