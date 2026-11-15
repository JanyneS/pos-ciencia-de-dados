# 🖼️ AT 04 — Convolutional Neural Network (CNN)

Atividade da disciplina de **Redes Neurais e Deep Learning**, da Pós-Graduação em Ciência de Dados — UTFPR.

**Tema:** Análise comparativa de arquiteturas de Redes Neurais Convolucionais: LeNet-5, AlexNet e ResNet

## 🎯 Objetivo

Pesquisar na literatura ao menos três arquiteturas de CNN usadas para classificação de imagens e compará-las quanto à tarefa, aos datasets, à topologia, ao algoritmo de treinamento, às métricas, à ferramenta de implementação e aos prós e contras de cada uma.

## 🔍 Arquiteturas analisadas

| | LeNet-5 (1998) | AlexNet (2012) | ResNet (2015) |
|---|---|---|---|
| **Tarefa** | Dígitos manuscritos (10 classes) | Objetos em imagens naturais (1000 classes) | Objetos (1000 classes) e base para detecção |
| **Dataset** | MNIST | ImageNet (ILSVRC) | ImageNet, CIFAR-10, COCO e VOC |
| **Profundidade** | 7 camadas | 8 camadas (5 conv. + 3 FC) | 18 a 152 camadas |
| **Parâmetros** | ~60 mil | ~60 milhões | ~25 a ~60 milhões |
| **Inovações** | Convolução e subamostragem com pesos compartilhados | ReLU, dropout, aumento de dados, treino em GPU | Blocos residuais, batch normalization, global average pooling |
| **Treinamento** | Gradiente descendente (diagonal Levenberg-Marquardt) | SGD com momentum 0,9 | SGD com momentum 0,9 |
| **Métricas** | Taxa de erro no teste | Erro top-1 e top-5 | Erro top-1 e top-5 |
| **Resultado** | 0,95% de erro | 15,3% de erro top-5 | 3,57% de erro top-5 |
| **Ferramenta** | Implementação própria (Bell Labs) | cuda-convnet (C++/CUDA) | Caffe |

## 💡 Principais conclusões

- A evolução das CNNs combinou **mais dados, mais profundidade e melhores técnicas de otimização**.
- A LeNet-5 resolve bem problemas simples, mas tem pouca capacidade para imagens naturais.
- A AlexNet mostrou o ganho de CNNs profundas com grandes datasets e GPU, e popularizou ReLU, dropout e aumento de dados.
- A ResNet superou a degradação de redes muito profundas com conexões residuais, permitindo mais de cem camadas com custo moderado.

## 📂 Arquivos

- `relatorio_at04_cnn.pdf` — relatório entregue, com a análise comparativa completa
