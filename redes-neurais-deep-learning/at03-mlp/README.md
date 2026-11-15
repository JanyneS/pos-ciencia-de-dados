# 🧠 AT 03 — Multilayer Perceptron (MLP)

Atividade da disciplina de **Redes Neurais e Deep Learning**, da Pós-Graduação em Ciência de Dados — UTFPR.

**Tema:** Avaliação de um Perceptron Multicamadas (MLP) no diagnóstico de câncer de mama

## 🎯 Objetivo

Escolher uma implementação de MLP da literatura, avaliar o seu desempenho em um dataset público e elaborar um relatório técnico sobre o dataset, a implementação, a metodologia experimental e os resultados.

## 📊 Dataset

**Breast Cancer Wisconsin (Diagnostic) — WDBC**, ID 1510 no [OpenML](https://www.openml.org/d/1510), originalmente do repositório UCI.

- 569 amostras e 30 atributos numéricos extraídos de imagens de núcleos celulares
- Classificação binária: benigno (357, 62,7%) ou maligno (212, 37,3%)
- Sem valores faltantes; atributos com escalas muito diferentes

## 🔧 Implementação e metodologia

- **Modelo:** `MLPClassifier` do scikit-learn, encadeado a um `StandardScaler` num `Pipeline`
- **Divisão:** holdout estratificado 70/30 (398 treino / 171 teste)
- **Seleção de hiperparâmetros:** validação cruzada estratificada 10-fold, só no treino
- **Métricas:** acurácia, F1 e recall da classe maligna; no teste, também precisão, ROC AUC e matriz de confusão

Análises realizadas:
1. Dados originais × normalizados
2. Número de neurônios (1 a 200) e redes com duas camadas ocultas
3. Taxa de aprendizagem (0,0001 a 1,0) com os otimizadores SGD e Adam
4. Avaliação final no teste, robustez à semente e comparação com regressão logística

## 📈 Principais resultados

**Modelo final:** StandardScaler + MLP com uma camada de 100 neurônios, Adam, taxa de aprendizagem 0,001.

| Modelo (teste) | Acurácia | Precisão | Recall | F1 | ROC AUC |
|----------------|----------|----------|--------|----|---------|
| MLP final (normalizado) | 0,971 | 1,000 | 0,922 | 0,959 | 0,994 |
| Regressão Logística (normalizada) | 0,971 | 0,984 | 0,938 | 0,960 | 0,998 |
| MLP sem normalização | 0,930 | 0,981 | 0,828 | 0,898 | 0,994 |

- A **normalização** foi o fator mais importante: sem ela, o número de tumores malignos não detectados mais que dobrou (11 contra 5).
- A partir de 10 neurônios, a arquitetura teve pouco efeito no desempenho.
- A **taxa de aprendizagem** foi decisiva para a convergência: o SGD dependeu muito dela, e o Adam ficou instável a partir de 0,1.

## 📂 Arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `relatorio_at03_mlp.pdf` | Relatório técnico entregue |
| `01_exploracao.py` | Análise exploratória e estatísticas do dataset |
| `02_normalizacao.py` | Comparação entre dados originais e normalizados |
| `03_neuronios.py` | Efeito do número de neurônios e de camadas ocultas |
| `04_taxa_aprendizagem.py` | Efeito da taxa de aprendizagem com SGD e Adam |
| `05_modelo_final.py` | Avaliação final no teste e comparação de modelos |
| `requirements.txt` | Dependências |

## ▶️ Como executar

```bash
pip install -r requirements.txt
python 01_exploracao.py
python 02_normalizacao.py
python 03_neuronios.py
python 04_taxa_aprendizagem.py
python 05_modelo_final.py
```

Os scripts baixam o dataset do OpenML e salvam as figuras em `figuras/` e as tabelas em `resultados/`.
