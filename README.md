# RNA-Power-Flow
Neural Network structured for power flow analysis

### Esse projeto tem como objetivo o estudo de redes neurais artificiais aplicadas em sistemas de energia elétrica.
Para análise foram utilizados dados reais de carga e sistemas IEEE com benchmarks conhecidos na literatura. 

No projeto também há instruções sobre como encontrar os melhores parametros para uma RNA desse tipo (Grid Search)

## Sistema de Estudo
[IEEE 14 Barras](https://github.com/user-attachments/files/22007384/Imagem1.tif)

### IEEE 14, 30, 58, 117 e 300 barras disponiveis em : 
https://labs.ece.uw.edu/pstca/

### Curvas de carga ONS disponiveis em :
https://dados.ons.org.br/dataset/curva-carga

A curva de carga utilizada no estudo foi a de 2023, filtrada para região sudeste

## Grid Search sistemático para encontrar melhores parâmetros para RNA
A busca dos melhores parametros foi feita combinando 
- o numero de camadas (1, 2, 3, 4 e 5)
- o numero de neuronios por camada (8, 16, 32, 64, 128, 256 e 512)
- funções de ativação (linear, relu, tanh, sigmoid)

### Dessa forma, foram criados 140 modelos que foram comparados entre si para escolher a melhor arquitetura. 
