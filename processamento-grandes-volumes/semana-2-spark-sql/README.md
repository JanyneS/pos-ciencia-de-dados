# ⚡ Semana 2 — Spark na prática (Spark SQL)

Atividade da disciplina de **Processamento de Grandes Volumes de Dados**, da Pós-Graduação em Ciência de Dados — UTFPR.

**Tema da semana:** Tecnologias de processamento de grandes volumes de dados  
**Período:** 31/10/2026 a 06/11/2026

## 🎯 Objetivo

Usar o **Apache Spark** (PySpark e Spark SQL) para consultar uma base de dados de churn de clientes de telefonia e responder perguntas com agrupamentos e filtros.

## ⚙️ Ambiente

- Google Colab
- Apache Spark 3.5.1 (PySpark), em modo local
- Dataset: `churn-bigml-20.csv` (667 registros após remover duplicados)

## 🧹 Preparação dos dados

Seguindo o notebook da prática: remoção de registros duplicados, renomeação de `Account length` para `Account`, remoção de `Number vmail messages` e criação da view temporária `temp`.

## 🔧 Consultas e resultados

### 1. Média de chamadas por código de área (`Total day calls`)

```python
df.groupBy('Area code').agg({'Total day calls': 'mean'}).orderBy('Area code').show()
```

| Area code | Média de chamadas |
|-----------|-------------------|
| 408 | 99,39 |
| 415 | 101,93 |
| 510 | 100,48 |

### 2. Média de minutos por código de área (`Total day minutes`)

```python
df.groupBy('Area code').agg({'Total day minutes': 'mean'}).orderBy('Area code').show()
```

| Area code | Média de minutos |
|-----------|------------------|
| 408 | 175,45 |
| 415 | 181,42 |
| 510 | 185,72 |

### 3. Registros com chamadas noturnas menores que 55 (`Total night calls < 55`)

```python
noturnas = df.filter(df['Total night calls'] < 55)
noturnas.show(truncate=False)
```

A consulta retornou **5 registros** (estados ID, AL, NH, MT e AL), com chamadas noturnas entre 42 e 53.

## 📂 Arquivos

- `atividade_spark_sql.pdf` — documento entregue, com os prints da execução
- `atividade_spark_sql_colab.ipynb` — notebook do Google Colab com as consultas executadas
