# PARTE 1: Preparar dados
# ----------------------------

X = np.load('X_normalizado.npy')
y = np.load('y_saida.npy')
scaler_y = joblib.load('scaler_y.pkl')
y_normalizado = scaler_y.transform(y)

# Carregar nomes das colunas de entrada a partir do CSV original
df = pd.read_csv('cenarios_ieee14_completos_com_fatores.csv')

# Tentativa de capturar colunas numéricas que compõem X
colunas_entrada = df.select_dtypes(include=[np.number]).columns.tolist()

# Ajustar para bater exatamente com X
if len(colunas_entrada) >= X.shape[1]:
    colunas_entrada = colunas_entrada[:X.shape[1]]
else:
    print(f"⚠️ Apenas {len(colunas_entrada)} nomes encontrados para {X.shape[1]} colunas em X.")
    colunas_entrada = [f'Var_{i}' for i in range(X.shape[1])]
    print("🔁 Usando nomes genéricos temporariamente.")

# ----------------------------
# PARTE 2: Treinar modelo RNA
# ----------------------------

modelo = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(y.shape[1], activation='linear')
])

modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("🎯 Treinando modelo vencedor...")
modelo.fit(X, y_normalizado, epochs=100, batch_size=32, verbose=0)
print("✅ Treinamento concluído.")

# ----------------------------
# PARTE 3: Inferência RNA
# ----------------------------

start_time = time.time()
y_pred_normalizado = modelo.predict(X)
nn_total_time = time.time() - start_time

y_pred = scaler_y.inverse_transform(y_pred_normalizado)

print(f"\n⏱️ Tempo total da RNA para {X.shape[0]} entradas: {nn_total_time:.4f} segundos")

# ----------------------------
# PARTE 4: Comparar tempo com solver
# ----------------------------

solver_total_time = 392.17  # substitua com o valor real medido

print(f"\n📊 Comparação de desempenho:")
print(f"- Solver tradicional: {solver_total_time:.2f} segundos")
print(f"- Rede neural (inferência): {nn_total_time:.2f} segundos")
print(f"- A RNA foi aproximadamente {solver_total_time / nn_total_time:.2f} vezes mais rápida")

# ----------------------------
# PARTE 5: Random Forest para importância das variáveis
# ----------------------------

print("\n🌲 Analisando importância das entradas com Random Forest...")

# Calcular erro absoluto da RNA por amostra
erro_abs = np.abs(y - y_pred)
erro_medio = erro_abs.mean(axis=1) if erro_abs.ndim > 1 else erro_abs

# Treinar Random Forest para prever erro da RNA com base nas entradas
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, erro_medio)

# Importâncias com nomes reais
importancias = rf.feature_importances_
df_importancia = pd.DataFrame({
    'Variavel': colunas_entrada,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

# Exibir top 10
print("\n🔎 Top 10 variáveis mais importantes na precisão da RNA:")
print(df_importancia.head(10))

# Plot
plt.figure(figsize=(14, 6))
plt.bar(df_importancia['Variavel'], df_importancia['Importancia'])
plt.xticks(rotation=90)
plt.xlabel('Variáveis de Entrada')
plt.ylabel('Importância Estimada')
plt.title('Importância das Variáveis de Entrada (Random Forest)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Salvar CSV
df_importancia.to_csv('importancia_variaveis.csv', index=False)
print("\n💾 Arquivo 'importancia_variaveis.csv' salvo com sucesso.")
