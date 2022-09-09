
# SVCf - Sistema de Visão Computacional para Frutas

O SVCf é um módulo para a identificação de pontos de corte em frutas
com pedúnculo.

## Funcionalidades

- Localiza a área do pedúnculo
- Gera pontos candidatos ao corte 
- Função que relaciona as distância dos pontos candidatos para a seleção do mais próximo da fruta


## Função coordenadaPontoFinal()

```http
  import PontoPedunculo
```

#### Entrada


| Parâmetro   | Tipo       | Descrição                           |
| :---------- | :--------- | :---------------------------------- |
| `areaPedunculo` | `array` | imagem da região do pedunculo |
| `limiarBaixo` | `array` | valores mínimo do vetor característica do pedúnculo |
| `limiarAlto` | `array` | valores máximo do vetor característica do pedúnculo |
| `topLeftX` | `int` | coordenada X da caixa da fruta |
| `topLeftY` | `int` | coordenada Y da caixa da fruta |
| `tipoBusca` | `string` | Ponderada - Media - Kmeans |
| `qtdPontos` | `int` | Quantidade de cluster  |

#### Retorna

| Parâmetro   | Tipo       | Descrição                                   |
| :---------- | :--------- | :------------------------------------------ |
| `coordenadaFinalX` | `int` | coordenada X |
| `coordenadaFinalY` | `int` | coordenada Y |
| `areaPedunculo` | `array` | imagem da região do pedunculo |
| `imagemHUE` | `array` | imagem com os pontos candidatos marcados |

Retorna as coordenada final do ponto de corte

## Utiliza

[OpenCV](https://opencv.org/)

[Numpy](https://numpy.org/)

[Matplotlib](https://matplotlib.org/)
