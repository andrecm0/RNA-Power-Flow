import pandapower.networks as pn
import pandapower as pp
import copy
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

######################################################################################################################

# Carrega o sistema IEEE XX barras
net = pn.case14() 
# net = pn.case30() // net = pn.case57() // net = pn.case118() // net = pn.case300()

# Roda o fluxo de potência no sistema
pp.runpp(net)

# Tensões nas barras
display(net.res_bus[['vm_pu', 'va_degree']])

# Carrega a curva de Carga
curva = pd.read_csv("Curva_de_Carga_Sudeste_2023.csv", parse_dates=['din_instante'])
curva.head()

# Normaliza a curva
curva['fator_carga'] = curva['carga_mw'] / curva['carga_mw'].max()

######################################################################################################################

# Gerar cenários e medir tempo
cenarios = []
start_solver = time.time()  # início da contagem de tempo no solver

# Gerar cenários

for i, row in curva.iterrows():
    net = copy.deepcopy(base_net)

    fator = row['fator_carga']
    net.load['p_mw'] *= fator
    net.load['q_mvar'] *= fator

    pp.runpp(net)

    entrada = {
        **{f'load_p_mw_{i}': p for i, p in enumerate(net.load['p_mw'])},
        **{f'load_q_mvar_{i}': q for i, q in enumerate(net.load['q_mvar'])},
        **{f'gen_vm_pu_{i}': v for i, v in enumerate(net.gen['vm_pu'])},
        'hora': row['hora'],
        'dia_semana': row['dia_semana'],
        'mes': row['mes']
    }

    saida = {
        **{f'bus_vm_pu_{i}': v for i, v in enumerate(net.res_bus['vm_pu'])},
    }

    cenario = {**entrada, **saida}
    cenarios.append(cenario)

solver_total_time = time.time() - start_solver  # fim da contagem
print(f"Tempo total solver para {len(curva)} casos: {solver_total_time:.2f} segundos")

# Criar DataFrame e salvar
cenarios_df = pd.DataFrame(cenarios)
cenarios_df.to_csv('cenarios_ieee14_completos_com_fatores.csv', index=False)
print("Cenários com fatores de carga salvos com sucesso!")

######################################################################################################################

# 1. Carregar os dados
df = pd.read_csv('cenarios_ieee14_completos_com_fatores.csv')

# 2. Separar entradas e saídas
colunas_entrada = [col for col in df.columns if (
    col.startswith('load_p_mw_') or
    col.startswith('load_q_mvar_') or
    col.startswith('gen_vm_pu_') or
    col in ['hora', 'dia_da_semana', 'mes']
)]

colunas_saida = [col for col in df.columns if
    col.startswith('bus_vm_pu_') ]

X = df[colunas_entrada].values
y = df[colunas_saida].values

# 3. Normalizar entradas (StandardScaler é recomendado para RNA com ReLU ou Linear)
scaler_X = StandardScaler()
X_normalizado = scaler_X.fit_transform(X)

# 4. Salvar os arrays e o scaler
np.save('X_normalizado.npy', X_normalizado)
np.save('y_saida.npy', y)
joblib.dump(scaler_X, 'scaler_X.pkl')

print("Pré-processamento concluído com sucesso. Dados prontos para treinar a RNA.")

######################################################################################################################

# 1. Carregar dados
X = np.load('X_normalizado.npy')
y = np.load('y_saida.npy')

# 2. Normalizar y
scaler_y = StandardScaler()
y_normalizado = scaler_y.fit_transform(y)

# 3. Salvar o scaler para uso posterior
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Scaler de y salvo como 'scaler_y.pkl'.")

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_normalizado, test_size=0.2, random_state=42)

# Modelo
modelo = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(y.shape[1], activation='linear')
])
modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
modelo.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0)

######################################################################################################################
  
# Inferência
y_pred_normalizado = modelo.predict(X_test)
y_previsto_original = scaler_y.inverse_transform(y_pred_normalizado)
y_test_original = scaler_y.inverse_transform(y_test)

# Gráfico: Previsões vs Valores Reais
plt.figure(figsize=(10,6))
plt.scatter(y_test_original, y_previsto_original, color='blue', alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         color='red', linestyle='--')
plt.title('Previsões vs. Valores Reais')
plt.xlabel('Valores Reais')
plt.ylabel('Previsões')
plt.grid(True)
plt.show()
