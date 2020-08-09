from flask import Flask
from driver import get_final_kmeans

app = Flask(__name__)

@app.route("/")
def hello():
    return get_final_kmeans()