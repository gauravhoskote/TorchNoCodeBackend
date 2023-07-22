from flask import Flask, request, jsonify
import json
app = Flask(__name__)

ACK_MESSAGE = {"response": "Acknowledged"}

@app.route('/')
def hello():
    return 'Hello, World!'



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