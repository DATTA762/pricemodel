from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained pipeline
with open("taxi_price_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        
            # Collect input from form
            trip_distance = float(request.form["Trip_Distance_km"])
            time_of_day = request.form["Time_of_Day"]
            day_of_week = request.form["Day_of_Week"]
            passenger_count = float(request.form["Passenger_Count"])
            traffic = request.form["Traffic_Conditions"]
            weather = request.form["Weather"]
            base_fare = float(request.form["Base_Fare"])
            per_km_rate = float(request.form["Per_Km_Rate"])
            per_minute_rate = float(request.form["Per_Minute_Rate"])
            duration = float(request.form["Trip_Duration_Minutes"])

            # Create a DataFrame with the exact column names as training
            input_df = pd.DataFrame([{
                "Trip_Distance_km": trip_distance,
                "Time_of_Day": time_of_day,
                "Day_of_Week": day_of_week,
                "Passenger_Count": passenger_count,
                "Traffic_Conditions": traffic,
                "Weather": weather,
                "Base_Fare": base_fare,
                "Per_Km_Rate": per_km_rate,
                "Per_Minute_Rate": per_minute_rate,
                "Trip_Duration_Minutes": duration
            }])

            # Make prediction
            pred = model.predict(input_df)[0]

            # If you log-transformed Trip_Price during training, reverse it:
            # pred = np.expm1(pred)

            prediction = f"Estimated Trip Price: ₹ {pred:.2f}"

        # except Exception as e:
        #     prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
