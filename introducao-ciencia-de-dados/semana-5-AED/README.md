# Semana 5 — Transformação dos Dados

Atividade desenvolvida na disciplina de **Introdução à Ciência de Dados**, da Pós-Graduação em Ciência de Dados — UTFPR.

## 🎯 Objetivo

Aplicar técnicas de transformação e preparação de dados, explorando normalização, agregação, criação de novos atributos e redução de registros.

## 🔧 Etapas realizadas

### Normalização
Aplicação de três métodos sobre as variáveis de notas:

- Normalização Simples
- MinMax
- Padronização (Z-score)

### Agregação
Agrupamento dos estudantes por curso para análise de:

- Média da nota de Matemática;
- Quantidade de estudantes por curso;
- Representação dos resultados por meio de gráficos de barras.

### Criação de novos atributos
Foram derivados novos atributos para enriquecer a análise:

- **Idade** — calculada a partir da data de nascimento;
- **Média_Geral** — calculada a partir das notas de Matemática, Programação e Estatística;
- **Faixa_Desempenho** — classificação dos estudantes em desempenho Alto, Médio ou Baixo.

### Redução dos dados
Identificação de registros duplicados utilizando a combinação:

`Nome + Data_Nascimento + Curso`

Os registros duplicados foram removidos mantendo apenas a primeira ocorrência.

## 🛠️ Tecnologias e conceitos

- Análise Exploratória de Dados (AED)
- Transformação de Dados
- Normalização
- MinMax
- Z-score
- Agregação de Dados
- Feature Engineering
- Criação de Atributos
- Tratamento de Duplicatas
- Visualização de Dados

## 📁 Arquivos

- `Atividade_semana_5.pdf` — relatório da atividade;
- `dataset_limpo.xlsx` — base resultante após as transformações realizadas.

