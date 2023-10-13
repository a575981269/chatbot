import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)

# 获取Heroku分配的端口，如果没有分配则默认使用5000端口
port = int(os.environ.get("PORT", 5000))

@app.route("/")
def index():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    # 使用从环境变量获取的端口号
    app.run(host="0.0.0.0", port=port)
