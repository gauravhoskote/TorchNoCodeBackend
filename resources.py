from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json
import time
import subprocess

app = Flask(__name__)
CORS(app)

ACK_MESSAGE = {"response": "Acknowledged"}

@app.route('/')
def hello():
    return 'Hello, World!'


def start_training():
    print("Training started")
    time.sleep(10)
    subprocess.run(["python", "torch_template.py"])



@app.route('/train', methods = ['POST'])
def train():
    try:
        request_body = request.json
        with open('parameters.json', 'w') as fp:
            json.dump(request_body, fp)
        return jsonify(ACK_MESSAGE)
    except Exception as exc:
        return "Something went wrong: " + str(exc)



if __name__ == '__main__':
    app.run(debug=True)