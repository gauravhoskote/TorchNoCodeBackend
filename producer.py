from kafka import KafkaProducer
import json

PUBSUB_TOPIC = 'train_models'
KAFKASERVER = 'localhost:9092'
producer = KafkaProducer(bootstrap_servers=KAFKASERVER)
mapper = {
    'a': 1,
    'b': 2
}

producer.send(PUBSUB_TOPIC, json.dumps(mapper).encode('utf-8'))
producer.flush()