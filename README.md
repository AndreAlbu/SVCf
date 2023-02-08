
# SVCf - Sistema de Visão Computacional para Frutas

O SVCf é um módulo para a identificação de pontos de corte em frutas
com pedúnculo.

## Funcionalidades

- Localiza a área do pedúnculo
- Gera pontos candidatos ao corte 
- Funções para encontrar o ponto de corte que utiliza:
  - Média ponderada
  - Média Aritmética 
  - Kmeans
  
## Modo de utilizar

Para utilizar o módulo faça o download do arquivo PC.py

Logo em seguida, coloque-o dentro da mesma pasta do arquivo da chamada principal.

```bash
  import PC as pc
```
Com isso, é possível ter acesso a todas a funções do arquivo.

## Uso

A etapa inicial é localizar a área do pedúnculo, para isso, utiliza-se a função de preve a área, na qual se baseia na área (caixa) da fruta.

```python
areaPedunculo = pc.preveAreaPedunculo(xt, yt, xb, yb, limiarLargura, limiarAltura, distanciaCaixa)
```

| Parâmetro   | Tipo       | Descrição                           |
| :---------- | :--------- | :---------------------------------- |
| `xt, yt` | `int` | coordenada superior da caixa da fruta |
| `xb, yb` | `int` | coordenada inferior da caixa da fruta |
| `limiarLargura` | `float` |largura desejada para a área do pedúnculo [0,1] |
| `limiarAltura` | `float` | altura desejada para a área do pedúnculo [0,1] |
| `distanciaCaixa` | `int` | distância entre a caixa do pedúnculo e a caixa da fruta |

#### Retorna as coordenadas da área do pedúnculo

```
  x1, y1, x2, y2 = areaPedunculo[0], areaPedunculo[1], areaPedunculo[2], areaPedunculo[3]
```

#### Área do pedúnculo

Com as coordenadas da área é realizado um corte para que seja obtido somente a parte da imagem referente a área do pedúnculo.

```
corteAreaPedunculo = imagem_pedunculo[y2:y1, x1:x2]
```

#### Localização da coordenada final do ponto de corte

```python
coordenadaFinal = pc.coordenadaPontoFinal(corteAreaPedunculo,
                                                  baixo,
                                                  alto,
                                                  TopLeftX, 
                                                  TopLeftY, 
                                                  " ", 
                                                  valor,
                                                  metodo)
```

| Parâmetro   | Tipo       | Descrição                           |
| :---------- | :--------- | :---------------------------------- |
| `areaPedunculo` | `array` | imagem da região do pedunculo |
| `limiarBaixo` | `array` | valores mínimo do vetor característica do pedúnculo |
| `limiarAlto` | `array` | valores máximo do vetor característica do pedúnculo |
| `topLeftX` | `int` | coordenada X da caixa da fruta |
| `topLeftY` | `int` | coordenada Y da caixa da fruta |
| `tipoBusca` | `string` | Ponderada - Media - Kmeans |
| `qtdPontos` | `int` | Quantidade de cluster  |
| `metodo` | `int` | Remove os outliners  |

Existe duas maneiras de remover os outliners: 1 (outliners) 2 (IsolationForest)

#### Retorna

```
  pontoX, pontoY, areaSegmentada, pontosCandidatos = coordenadaFinal[0], coordenadaFinal[1], coordenadaFinal[2], coordenadaFinal[3]
```

| Parâmetro   | Tipo       | Descrição                                   |
| :---------- | :--------- | :------------------------------------------ |
| `coordenadaFinalX` | `int` | coordenada X |
| `coordenadaFinalY` | `int` | coordenada Y |
| `areaPedunculo` | `array` | imagem da região do pedunculo |
| `imagemHUE` | `array` | imagem com os pontos candidatos marcados |


# Banco de Imagens

Local de captura: Instituto Federal de Educação Ciência e Tecnologia do Ceará, campus Crato. 

Dia: 09 de Novembro de 2022

Hora: 15:30 - 17:30

Aparelho de captura: Poco X3 Pro

Dimensões: 3000x3000

**************
### Imagens de treinamento redimensionadas para 500x500
Imagens para treinamento: 535

### Imagens de validação redimensionadas para 1500x1500

Composta: 103

Inclinada: 12

Oclusa Manga: 149

Oclusa Pedúnculo: 169

Simples com Folhas no Fundo: 208

Simples sem Folhas no Fundo: 249
 

Link para acessar a base de imagens: 

[Kaggle](https://www.kaggle.com/datasets/andreifce/recognition-mango)

## Melhorias

- Modularização do código
- Adição do método K-means para encontrar a coordenada final
- Definição do tempo de busca para encontrar a fruta

## Utiliza

[OpenCV](https://opencv.org/)

[Numpy](https://numpy.org/)

[Matplotlib](https://matplotlib.org/)
