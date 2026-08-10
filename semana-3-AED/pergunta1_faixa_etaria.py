# ============================================================
# PERGUNTA 1: Existe uma faixa etária predominante entre os evadidos?
# Variáveis: Idade (quantitativa discreta) + Status (qualitativa nominal)
# Medidas usadas:
#   - Posição: média, mediana, moda
#   - Dispersão: desvio padrão, variância
# ============================================================

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("dados.csv")

evadidos = df[df["Status"] == "Evadido"]

idades = evadidos["Idade"]

# --- 3. Medidas de Posição ---

# Média: 
media = idades.mean()

# Mediana:
mediana = idades.median()

# Moda:
moda = idades.mode()[0]  

# --- 4. Medidas de Dispersão ---

# Desvio Padrão: quanto as idades se afastam da média em média
desvio_padrao = idades.std()

variancia = idades.var()


print("=" * 50)
print("PERGUNTA 1: Faixa etária predominante dos evadidos")
print("=" * 50)
print(f"Total de evadidos analisados: {len(idades)}")
print()
print("--- Medidas de Posição ---")
print(f"Média:   {media:.2f} anos")
print(f"Mediana: {mediana:.2f} anos")
print(f"Moda:    {moda} anos")
print()
print("--- Medidas de Dispersão ---")
print(f"Desvio Padrão: {desvio_padrao:.2f} anos")
print(f"Variância:     {variancia:.2f} anos²")
print()

print("--- Interpretação ---")
print(f"A idade média dos evadidos é {media:.1f} anos.")
print(f"A idade mais comum (moda) entre os evadidos é {moda} anos.")
print(f"O desvio padrão de {desvio_padrao:.1f} anos indica que as idades")
print(f"variam em média {desvio_padrao:.1f} anos em torno da média.")
