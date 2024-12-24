import sqlite3
from flask import Flask, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/measurements')
def get_measurements():
    conn = sqlite3.connect('measurement.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, heartRate, spo2 FROM measurements ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    # Transformar em dicionário para retornar como JSON
    # Algo do tipo: [{"id": 1, "timestamp": ..., "heartRate":..., "spo2":...}, ...]
    measurements = []
    for row in data:
        measurements.append({
            "id": row[0],
            "timestamp": row[1],
            "heartRate": row[2],
            "spo2": row[3]
        })

    return jsonify(measurements)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
