# 🔁 AT 05 — Autoencoders (AE)

Atividade da disciplina de **Redes Neurais e Deep Learning**, da Pós-Graduação em Ciência de Dados — UTFPR.

**Tema:** Pré-treinamento de um classificador de imagens com um Denoising Autoencoder (DAE)

## 🎯 Objetivo

Compreender os principais tópicos teóricos sobre Autoencoders e realizar um exercício prático: usar um **Denoising Autoencoder (DAE)** para pré-treinar um classificador de dígitos do MNIST e verificar se a rede tem desempenho melhor com ou sem essa etapa quando há apenas 500 imagens rotuladas.

## 📝 Exercícios teóricos

1. Principais tarefas em que os Autoencoders podem ser aplicados
2. Como os AEs ajudam quando há muitos dados não rotulados e poucos rotulados
3. Se uma reconstrução perfeita indica um bom AE, e como avaliar um AE
4. AEs subcompletos e sobrecompletos e os riscos de cada um
5. Pesos amarrados em um Stacked Autoencoder (SAE)

## 🔧 Exercício prático

| Item | Escolha |
|------|---------|
| Dados | MNIST do Keras (60.000 treino / 10.000 teste), pixels em [0, 1] |
| Ruído | Gaussiano (σ = 0,3), recortado em [0, 1] |
| DAE | Densa: 784 → 256 → 128 → 256 → 784 (ReLU; saída sigmoide), entropia cruzada binária, Adam |
| Classificador | Encoder do DAE (784 → 256 → 128) + Dense(10, softmax) |
| Treino do classificador | 500 imagens aleatórias (80% treino / 20% validação), early stopping |
| Métricas | Acurácia, F1-macro e matriz de confusão no teste |
| Repetições | 5 sorteios diferentes das 500 imagens |

O DAE reduziu o erro das imagens em relação às originais de **MSE 0,0467** (com ruído) para **0,0082** (reconstruídas).

## 📈 Resultados (teste, média ± desvio de 5 repetições)

| Cenário | Acurácia (%) | F1-macro (%) |
|---------|--------------|--------------|
| **Sem pré-treinamento** | **86,32 ± 0,73** | **86,09 ± 0,78** |
| DAE congelado | 83,54 ± 1,43 | 83,25 ± 1,50 |
| DAE + fine-tuning | 84,96 ± 1,14 | 84,68 ± 1,16 |

## 💡 Conclusão

Neste experimento, o pré-treinamento com o DAE **não melhorou** o desempenho: a rede treinada do zero foi a melhor. Entre os motivos discutidos no relatório estão a simplicidade do MNIST, o fato de o DAE aprender características voltadas para reconstrução (e não para separar classes) e o uso da mesma taxa de aprendizado no fine-tuning. Um DAE convolucional, uma taxa menor no fine-tuning ou menos exemplos rotulados poderiam tornar a vantagem do pré-treinamento mais evidente.

## 📂 Arquivos

- `relatorio_autoencoders.pdf` — relatório entregue, com os exercícios teóricos e a discussão crítica
- `autoencoders_dae.ipynb` — notebook do Google Colab com o código e as saídas
- [Notebook no Google Colab](https://colab.research.google.com/drive/1TDwOaidp4H0Y4PJwqdEW5ur9C9SW6dEO)
