from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load Model and Data Columns
with open("artifacts/columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]
    locations = data_columns[3:]  # Extract location names

with open("artifacts/banglore_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)


@app.route("/get_location_names", methods=["GET"])
def get_location_names():
    """Return a list of locations"""
    response = jsonify({"locations": locations})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route("/predict_home_price", methods=["POST"])
def predict_home_price():
    """Predict house price based on user input"""
    try:
        data = request.form
        total_sqft = float(data["total_sqft"])
        bhk = int(data["size_numeric"])
        bath = int(data["bath"])
        location = data["location"]

        # Prepare model input
        loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

        estimated_price = round(model.predict([x])[0], 2)

        return jsonify({"estimated_price": estimated_price})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(debug=True)
