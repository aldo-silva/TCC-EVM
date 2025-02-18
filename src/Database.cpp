#include "Database.hpp"
#include <sqlite3.h>
#include <iostream>

Database::Database()
    : m_db(nullptr) {
}

Database::~Database() {
    close();
}

bool Database::open(const std::string& dbName) {
    int rc = sqlite3_open(dbName.c_str(), reinterpret_cast<sqlite3**>(&m_db));
    if (rc != SQLITE_OK) {
        std::cerr << "Erro ao abrir/criar banco de dados: " << sqlite3_errmsg(reinterpret_cast<sqlite3*>(m_db)) << std::endl;
        m_db = nullptr;
        return false;
    }

    return true;
}

void Database::close() {
    if (m_db) {
        sqlite3_close(reinterpret_cast<sqlite3*>(m_db));
        m_db = nullptr;
    }
}

bool Database::createTable() {
    if (!m_db) {
        std::cerr << "Banco de dados não está aberto!" << std::endl;
        return false;
    }

    // Cria uma tabela de exemplo com:
    //  - id autoincrement
    //  - data/hora (timestamp) com valor padrão CURRENT_TIMESTAMP
    //  - heartRate (REAL)
    //  - spo2 (REAL)
    const char* sqlCreateTable =
        "CREATE TABLE IF NOT EXISTS measurements ("
        "   id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,"
        "   heartRate REAL,"
        "   spo2 REAL,"
        "   framePath TEXT"
        ");";

    char* errMsg = nullptr;
    int rc = sqlite3_exec(reinterpret_cast<sqlite3*>(m_db), sqlCreateTable, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "Erro ao criar tabela: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }

    return true;
}

bool Database::insertMeasurement(double heartRate, double spo2, const std::string& framePath) {
    if (!m_db) {
        std::cerr << "Banco de dados não está aberto!" << std::endl;
        return false;
    }

    // Prepara a declaração SQL com parâmetros para evitar injeção de SQL
    const char* sqlInsert = "INSERT INTO measurements (heartRate, spo2, framePath) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;

    int rc = sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(m_db), sqlInsert, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Erro ao preparar declaração SQL: " << sqlite3_errmsg(reinterpret_cast<sqlite3*>(m_db)) << std::endl;
        return false;
    }

    // Bind dos parâmetros
    sqlite3_bind_double(stmt, 1, heartRate);
    sqlite3_bind_double(stmt, 2, spo2);
    sqlite3_bind_text(stmt, 3, framePath.c_str(), -1, SQLITE_TRANSIENT);
    
    // Executa a declaração
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        std::cerr << "Erro ao inserir medição: " << sqlite3_errmsg(reinterpret_cast<sqlite3*>(m_db)) << std::endl;
        sqlite3_finalize(stmt);
        return false;
    }

    // Finaliza a declaração
    sqlite3_finalize(stmt);

    return true;
}
