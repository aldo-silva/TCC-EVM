#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <string>

// Classe de encapsulamento de operações com SQLite
class Database {
public:
    // Construtor / Destrutor
    Database();
    ~Database();

    // Abre (ou cria) o banco de dados
    bool open(const std::string& dbName);

    // Fecha o banco de dados
    void close();

    // Cria tabela de medições (caso não exista)
    bool createTable();

    // Insere uma medição de HR e SpO2
    bool insertMeasurement(double heartRate, double spo2, const framePath);

private:
    // Ponteiro para a instância do SQLite
    void* m_db;  // Usamos void* para evitar precisar incluir sqlite3.h no .hpp
};

#endif
