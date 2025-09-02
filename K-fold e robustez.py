# PARTE 6: K-Fold Cross Validation
# ----------------------------

print("\n🔁 Iniciando validação cruzada (K-Fold)...")

k = 5  # ou outro valor desejado
kf = KFold(n_splits=k, shuffle=True, random_state=42)

maes, rmses, r2s = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    print(f"\n📦 Fold {fold}/{k}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_normalizado[train_idx], y_normalizado[test_idx]

    # Novo modelo com a mesma arquitetura original
    modelo_fold = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(y.shape[1], activation='linear')
    ])
    modelo_fold.compile(optimizer='adam', loss='mse')

    modelo_fold.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Inferência
    y_pred_norm = modelo_fold.predict(X_test, verbose=0)

    # Desnormalizar
    y_pred_real = scaler_y.inverse_transform(y_pred_norm)
    y_test_real = scaler_y.inverse_transform(y_test)

    # Métricas
    mae = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)

    print(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.4f}")

    maes.append(mae)
    rmses.append(rmse)
    r2s.append(r2)

# ----------------------------
# Resultado final
# ----------------------------

print("\n📊 Resultados médios após K-Fold:")
print(f"MAE  médio: {np.mean(maes):.6f} ± {np.std(maes):.6f}")
print(f"RMSE médio: {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")
print(f"R²   médio: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
