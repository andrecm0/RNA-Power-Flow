# -------------------------------
# 1) Carregar dados
# -------------------------------
X = np.load('X_normalizado.npy')
y = np.load('y_saida.npy')

# Garantir que y seja 2D: (n amostras, n_saidas)
if y.ndim == 1:
    y = y.reshape(-1, 1)

# -------------------------------
# 2) Normalizar y e salvar scaler
# -------------------------------
scaler_y = StandardScaler()
y_normalizado = scaler_y.fit_transform(y)
joblib.dump(scaler_y, 'scaler_y.pkl')
print("Scaler de y salvo como 'scaler_y.pkl'.")

# -------------------------------
# 3) Split
# -------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y_normalizado, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42
)

# (opcional) salvar conjuntos
np.save('X_train.npy', X_train)
np.save('X_val.npy',   X_val)
np.save('X_test.npy',  X_test)
np.save('y_train.npy', y_train)
np.save('y_val.npy',   y_val)
np.save('y_test.npy',  y_test)
print("Conjuntos de dados salvos com sucesso.")

# --------------------------------------------------
# → Função para criar modelo
# --------------------------------------------------
def criar_modelo(n_neuronios, camadas, ativacao, input_dim, output_dim, dropout=0.0):
    modelo = Sequential()
    modelo.add(Input(shape=(input_dim,)))
    for _ in range(camadas):
        modelo.add(Dense(n_neuronios, activation=ativacao))
        if dropout > 0:
            modelo.add(Dropout(dropout))
    modelo.add(Dense(output_dim, activation='linear'))
    modelo.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return modelo

# --------------------------------------------------
# → GRID SEARCH (treino com MAE; métricas extras)
# --------------------------------------------------
ativacoes  = ['linear', 'relu', 'tanh', 'sigmoid']
neurons    = [8, 16, 32, 64, 128, 256, 512]
num_layers = [1, 2, 3, 4, 5]

resultados = []
best_mae_real = np.inf
best_model = None
scaler_y = joblib.load('scaler_y.pkl')

for ativacao, n_neuronios, camadas in itertools.product(ativacoes, neurons, num_layers):
    print(f"\n🔧 Testando: ativação={ativacao}, neurônios={n_neuronios}, camadas={camadas}")

    modelo = criar_modelo(n_neuronios, camadas, ativacao, X_train.shape[1], y_train.shape[1], dropout=0.0)

    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=[es],
        verbose=0
    )

    # Avaliação em escala normalizada
    _, mae_norm = modelo.evaluate(X_test, y_test, verbose=0)

    # Predição e métricas em ESCALA REAL (desnormalizada)
    y_pred_norm = modelo.predict(X_test, verbose=0)
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)
    y_test_real = scaler_y.inverse_transform(y_test)

    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real, multioutput='uniform_average')

    print(f"MAE (norm): {mae_norm:.6f} | MAE (real): {mae_real:.6f} | RMSE (real): {rmse_real:.6f}")

    resultados.append({
        'ativacao': ativacao,
        'neurônios': n_neuronios,
        'camadas': camadas,
        'mae_normalizado': mae_norm,
        'mae_real': mae_real,
        'rmse_real': rmse_real,
    })



# DataFrame e ordenação
df = pd.DataFrame(resultados)

# Ordenar por MAE real, depois RMSE, depois R² (em ordem decrescente no R²)
df = df.sort_values(by=['mae_real', 'rmse_real'], ascending=[True, True])

print("\n📊 Top 10 modelos (por MAE real, desempate por RMSE e R²):")
print(df.head(10))

# Salvar resultados e modelo
df.to_csv('resultados_comparativos.csv', index=False)
print("Resultados salvos em 'resultados_comparativos.csv'.")

if best_model is not None:
    best_model.save('melhor_modelo.h5')
    print(f"Melhor modelo salvo como 'melhor_modelo.h5' (MAE real = {best_mae_real:.6f}).")
