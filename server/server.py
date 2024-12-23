import sqlite3
from flask import Flask, render_template

app = Flask(__name__)

# Rota principal ("/")
@app.route('/')
def index():
    # Conecta ao banco de dados
    conn = sqlite3.connect('measurement.db')
    cursor = conn.cursor()
    
    # Consulta todos os registros da tabela measurements
    # Supondo que sua tabela seja: (id, timestamp, heartRate, spo2)
    cursor.execute("SELECT id, timestamp, heartRate, spo2 FROM measurements ORDER BY id DESC")
    data = cursor.fetchall()  # data será uma lista de tuplas
    conn.close()

    # data terá o formato [(id, timestamp, hr, spo2), (id, timestamp, hr, spo2), ...]
    # Renderiza o template passando 'data' como contexto
    return render_template('index.html', measurements=data)

if __name__ == "__main__":
    # Rode o servidor em 0.0.0.0 para ser acessível a todos na rede
    # Porta padrão 5000 (pode mudar se preferir)
    app.run(host='0.0.0.0', port=5000, debug=True)
