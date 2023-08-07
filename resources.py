from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json
import os
from torch_template import create_model
from kafka import KafkaProducer

PUBSUB_TOPIC = 'train_models'
app = Flask(__name__)
CORS(app)
thread_pool = []
ACK_MESSAGE = {"response": "Acknowledged"}
KAFKASERVER = 'localhost:9092'
producer = KafkaProducer(bootstrap_servers=KAFKASERVER)
@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/train', methods = ['POST'])
def train():
    try:
        request_body = request.json
        producer.send(PUBSUB_TOPIC, json.dumps(request_body).encode('utf-8'))
        producer.flush()
        return jsonify(ACK_MESSAGE)
    except Exception as exc:
        return "Something went wrong: " + str(exc)

if __name__ == '__main__':
    app.run(port=8000, debug=True, threaded=True)