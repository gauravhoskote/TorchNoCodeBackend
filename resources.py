from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import json
import os
from torch_template import create_model
from kafka import KafkaProducer
from database import get_connection, close_connection, create_table

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

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    first_name = data.get("first_name")
    last_name = data.get("last_name")
    email = data.get("email")
    occupation = data.get("occupation")
    username = data.get("username")
    password = data.get("password")

    # Validate all the fields (implement your validation logic here)

    try:
        conn = get_connection()  # Function to get a database connection
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO User_table (first_name, last_name, email, occupation, username, password) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (first_name, last_name, email, occupation, username, password),
        )
        conn.commit()
        close_connection(conn)  # Function to close the database connection

        return jsonify({"message": "Signup successful"}), 201
    except Exception as e:
        return jsonify({"message": "Signup failed", "error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    try:
        conn = get_connection()  # Function to get a database connection
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM User_table WHERE username = %s AND password = %s",
            (username, password),
        )
        user = cursor.fetchone()
        close_connection(conn)  # Function to close the database connection

        if user:
            return jsonify({"message": "Login successful"}), 200
        else:
            return jsonify({"message": "Invalid credentials"}), 401
    except Exception as e:
        return jsonify({"message": "Login failed", "error": str(e)}), 500


@app.route("/users", methods=["GET"])
def get_all_users():
    try:
        conn = get_connection()  # Function to get a database connection
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM User_table;")
        users = cursor.fetchall()
        close_connection(conn)  # Function to close the database connection

        # Convert the users data to a list of dictionaries for JSON serialization
        user_list = []
        for user in users:
            user_dict = {
                "user_id": user[0],
                "first_name": user[1],
                "last_name": user[2],
                "email": user[3],
                "occupation": user[4],
                "username": user[5],
                "password": user[6],
            }
            user_list.append(user_dict)

        return jsonify({"users": user_list}), 200
    except Exception as e:
        return jsonify({"message": "Failed to retrieve users", "error": str(e)}), 500


@app.route("/delete_all", methods=["DELETE"])
def delete_all_users():
    try:
        conn = get_connection()  # Function to get a database connection
        cursor = conn.cursor()
        cursor.execute("DELETE FROM User_table;")
        conn.commit()
        close_connection(conn)  # Function to close the database connection

        return jsonify({"message": "All records deleted successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Deletion failed", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8000, debug=True, threaded=True)