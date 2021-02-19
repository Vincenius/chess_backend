from flask import Flask
from flask import request, jsonify

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def new_user():
    image_data = request.files['input_name']
    # add here the code to create the user

    print(image_data)

    res = {"status": "ok"}
    return jsonify(res)