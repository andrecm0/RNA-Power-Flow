# RNAs Aplicadas à Engenharia Elétrica

## Esse projeto tem como objetivo o estudo de redes neurais artificiais aplicadas em sistemas de energia elétrica.
Para análise foram utilizados dados reais de carga e sistemas IEEE com benchmarks conhecidos na literatura. No projeto também há instruções sobre como encontrar os melhores parametros para uma RNA desse tipo (Grid Search)

### Sistema de Estudo
[IEEE 14 Barras](https://github.com/user-attachments/files/22007384/Imagem1.tif)
### IEEE 14, 30, 58, 117 e 300 barras disponiveis em : 
https://labs.ece.uw.edu/pstca/
### Curvas de carga ONS disponiveis em :
https://dados.ons.org.br/dataset/curva-carga

# A metodologia combina:  
- **Dados reais** de curva de carga (ONS – Região Sudeste, 2023).  
- **Simulações elétricas** em sistemas IEEE (14 barras, expansível para 30, 57, 118 e 300).  
- **Aprendizado supervisionado** para mapear cenários de operação em variáveis elétricas.  

O objetivo é demonstrar que uma RNA pode aprender padrões de fluxo de potência com **alta acurácia**, oferecendo uma alternativa rápida e escalável para estudos de operação.  

## Estrutura do Projeto
- `Curva_de_Carga_Sudeste_2023.csv` → base de dados de entrada (ONS).  
- `TG_André.ipynb` → notebook principal com pipeline completo.  
- Arquivos auxiliares → cenários, grid search, k-fold, random forest e bibliotecas.  

### Fluxo de execução:
1. **Geração dos cenários** → integração da curva de carga com sistemas IEEE.  
2. **Pré-processamento** → normalização (StandardScaler), treino/teste e K-Fold.  
3. **Modelagem** → RNA (Keras/TensorFlow) e Random Forest (baseline).  
4. **Validação** → métricas (MAE, RMSE).  
5. **Resultados** → gráficos e interpretação da performance.  

## Como Executar

Baixe/clone o repositório.
Abra o notebook principal:
jupyter notebook TG_André.ipynb

### Execute as células em sequência:

1- Importação da curva de carga. 

2- Geração de cenários.

3- Grid search para encontrar melhores parametros da RNA. 

4- Random forest para analise de features.  

5- k-fold para validação cruzada  

6- Cenário teste de generalização.

## Resultados Esperados

### Sistema IEEE 14 Barras

- **Rede Neural Artificial (RNA)**
  - MAE ≈ `0.000162`
  - RMSE ≈ `0.000375`
  - **Tempo de inferência**: cerca de **300 vezes mais rápido** que o solver tradicional (`pp.runpp` do Pandapower)


### Comparação de Desempenho

| Modelo              | MAE       | RMSE      | Velocidade  |
|---------------------|-----------|-----------|-------------------------|
| Solver (Pandapower) | -         | -         |   1x (referência)         |
| RNA (Keras)         | 0.0004    | 0.0010    | **≈ 300x mais rápido**  |

### Visualizações
- **Curvas de treinamento** → perda vs épocas (mostrando convergência estável da RNA).  
- **Comparação real vs predito** → tensões nas barras (a RNA reproduz fielmente os valores).  
- **Importância das variáveis** → ranking das features obtido pelo Random Forest, auxiliando na interpretabilidade.  

 **Resumo:**  
A RNA não apenas atinge **alta precisão**, mas também é **ordens de magnitude mais eficiente** em tempo de execução. Isso abre caminho para aplicações em **tempo real** e em **sistemas de maior porte**, onde a velocidade do cálculo de fluxo de potência é crítica.
