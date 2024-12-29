import sqlite3
from flask import Flask, render_template, jsonify, send_from_directory, abort

app = Flask(__name__)

CAPTURES_DIR = "/home/aldo/data/captures"

@app.route('/')
def index():
    # Renderiza o "index.html"
    return render_template('index.html')

@app.route('/api/measurements')
def get_measurements():
    conn = sqlite3.connect('measurement.db')
    cursor = conn.cursor()
    
    # Pegamos as 50 medições mais recentes (ORDER BY id DESC LIMIT 50)
    cursor.execute("""
        SELECT id, timestamp, heartRate, spo2, framePath
        FROM measurements
        ORDER BY id DESC
        LIMIT 50
    """)
    data = cursor.fetchall()
    conn.close()

    # Monta a lista de dicionários
    measurements = []
    for row in data:
        measurements.append({
            "id": row[0],
            "timestamp": row[1],
            "heartRate": row[2],
            "spo2": row[3],
            "imagePath": row[4]
        })

    # Retorna em formato JSON
    return jsonify(measurements)

@app.route('/captures/<filename>')
def serve_captures(filename):
    try:
        return send_from_directory(CAPTURES_DIR, filename)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
