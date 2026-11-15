# 🗄️ Semana 1 — Prática com HDFS

Atividade da disciplina de **Processamento de Grandes Volumes de Dados**, da Pós-Graduação em Ciência de Dados — UTFPR.

**Tema da semana:** Sistemas distribuídos e tecnologias de armazenamento de grandes volumes de dados  
**Período:** 24/10/2026 a 30/10/2026

## 🎯 Objetivo

Praticar os comandos básicos do **HDFS (Hadoop Distributed File System)** para criar diretórios, enviar arquivos, listar conteúdo e remover diretórios.

## ⚙️ Ambiente

- Google Colab
- Apache Hadoop 3.3.0
- Java 11 (OpenJDK)

## 🔧 Etapas realizadas

| # | Tarefa | Comando |
|---|--------|---------|
| 1 | Criar um diretório com o primeiro nome | `hadoop fs -mkdir Janyne` |
| 2 | Criar um diretório com o sobrenome | `hadoop fs -mkdir Soares` |
| 3 | Enviar o arquivo `california_housing_test.csv` para o diretório do primeiro nome | `hadoop fs -put sample_data/california_housing_test.csv Janyne/` |
| 4 | Listar o conteúdo do diretório | `hadoop fs -ls Janyne` |
| 5 | Excluir o diretório do sobrenome | `hadoop fs -rmdir Soares` |

Ao final, a listagem com `hadoop fs -ls` mostra que o diretório `Janyne` permanece com o arquivo enviado e que o diretório `Soares` foi removido.

## 📂 Arquivos

- `atividade_pratica_hdfs.pdf` — documento entregue, com os prints da execução
- `atividade_hdfs_colab.ipynb` — notebook do Google Colab com os comandos executados
