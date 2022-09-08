import cv2
import numpy as np
from matplotlib import pyplot as plt


def preveAreaPedunculo(xt, yt, xb, yb, limiarLargura, limiarAltura):
    
    """
    Função: Localizar a área do pedúnculo

    Parâmetro: xt, yt, xb, yb -> correspondem as coordenadas da caixa da fruta
               limiarLargura, limiarAltura -> determina a área do pedúnculo

    Ajuste: Os limiares podem ser ajustado no intervalo de 0 a 1
    """
    
    #Tamanho da região de interesse baseada na caixa da fruta
    Lmax = abs(xt - xb)
    Hmax = abs(yt - yb)

    RoiL = limiarLargura * Lmax
    RoiH = limiarAltura  * Hmax

    #Posiciona a área do pedúnculo no centro da caixa da fruta
    centro = (xb + xt) / 2

    centroCaixa = RoiL / 2

    x1 = centro - centroCaixa
    x2 = centro + centroCaixa

    y1 = abs(yt - 50) 
    y2 = abs(RoiH - y1)

    #Converte todas as coordenadas para um número inteiro
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    
    return x1, y1, x2, y2  

def histogramaHSV(image):

    """
    Função: Encontra o maior valor do histograma na componente HUE

    Parâmetro: Imagem do pedúnculo

    Ajuste: O valo mínimo utilizado é 1 e não 0, pois em uma imagem segmentada 
    a quantidade de 0 é muito superior comparada com as outras.

    """
    
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Hue
    histr0 = cv2.calcHist([image], [0], None, [180], [1, 180])
    
    # Obtém o maior valor do histograma
    ymax = max(histr0)
    xmax, _ = np.where(histr0 == ymax)

    valorHue = int(xmax)
         
    # Quantidade de pixel - posição    
    return valorHue

def segmentaAreaPedunculo(areaPedunculo, limiarBaixo, limiarAlto):

	"""
	Função: Segmenta a área do pedúnculo

	Parâmetro: areaPedunculo -> imagem da região do pedunculo (RGB)
	       hMIN, sMIN, vMIN -> limiares mínimos do HSV
	       hMAX, sMAX, vMAX -> limiares máximos do HSV

	Ajuste: Os limiares dependem da cultura em análise
	"""

	hMIN, sMIN, vMIN = limiarBaixo[0], limiarBaixo[1], limiarBaixo[2]
	hMAX, sMAX, vMAX = limiarAlto[0], limiarAlto[1], limiarAlto[2]

	baixo = np.array([hMIN, sMIN, vMIN])
	alto  = np.array([hMAX, sMAX, vMAX])

	img_hsv = cv2.cvtColor(areaPedunculo, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(img_hsv, baixo, alto)

	areaSegmentada = cv2.bitwise_and(areaPedunculo, areaPedunculo, mask = mask)

	return areaSegmentada

def encontraPontosCandidatos(areaPedunculo):

    """
    Função: Encontra os pontos candidatos ao corte

    Parâmetro: areaPedunculo -> imagem da região do pedunculo
    """

    areaPedunculo = cv2.cvtColor(areaPedunculo, cv2.COLOR_BGR2HSV)

    valorMaximoHUE = histogramaHSV(areaPedunculo)
        
    largura = areaPedunculo.shape[1]
    altura  = areaPedunculo.shape[0]
    
    centroX = round(largura / 2)
    centroY = round(altura / 2)
    
    todas_coordenadas  =  [] #Guarda todas as coordenadas (eixo X e Y)
    todas_coordenadasX =  [] #Guarda as coordenadas do eixo X vindas dos cantos encontrados
    todas_coordenadasY =  [] #Guarda as coordenadas do eixo Y vindas dos cantos encontrados
        
    for x in range(0, largura):

        for y in range(0, altura):

            h, s, v = areaPedunculo[y, x]

            if(h == valorMaximoHUE):

                todas_coordenadas.append((x,y))
                todas_coordenadasX.append(int(x))
                todas_coordenadasY.append(int(y))

                cv2.circle(areaPedunculo, (x, y), 1, (255, 0, 255), -1) #purple
            
    quantidadePontosEncontrados = len(todas_coordenadas)
    
    if(quantidadePontosEncontrados >= 1):
        
        return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo
    
    else:       
        
        todas_coordenadas, todas_coordenadasX, todas_coordenadasY = [(centroX, centroY)], [(centroX)], [(centroY)]
        
        return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo

    #cv2.imwrite("segmentada.png", areaPedunculo)
    
def calculaPesoY(pesosY):
    
    calPeso = (pesosY[0] * pesosY[0]) + (pesosY[1] * pesosY[1])
    
    calPeso = np.sqrt(calPeso)
    
    calPeso = int(calPeso)
    
    return calPeso

def calculaPesoX(pesosX, centro):
    
    #Exemplo: pesosX  -> 30
    #         centro  -> 25
    #         dif     -> 25 - 30
    #         centroX -> 30 + (-5)
    #         centroX = 25

    if(pesosX[0] > centro):
        
        dif = (centro - pesosX[0])
        
        centroX = pesosX[0] + dif
        
    else:
        
        centroX = pesosX[0]
        
    calPeso = (centroX * centroX) + (pesosX[1] * pesosX[1])
    
    calPeso = np.sqrt(calPeso)
    
    calPeso = round(calPeso)
    
    return calPeso

def combinacao(peso: list, todas_coordenadas: list) -> list:

    """
    Função: Fazer a combinação entre a coordenadas e os pesos encontrados
    """
    assert len(peso) == len(todas_coordenadas)
 
    quantidadePesos = len(peso)
    
    combina = []
 
    for i in range(quantidadePesos):
        
        combina.append((peso[i], todas_coordenadas[i]))
         
    return combina

def calculaMediaPontos(pontos):

    #Calcula a media dos pontos

    calMedia = (pontos[0] *  pontos[1])
    
    calMedia = int(calMedia)
    
    return calMedia

def encontraCoordenadasPonderada(areaPedunculo):

    """
    Função: Encontrar a coordenada (x, y) referente ao ponto de corte na área do pedúnculo utilizando
    a média ponderada dos pontos candidatos

    Parâmetro: areaPedunculo -> imagem da região do pedunculo
               todasCoordenadas, coordenadasX, coordenadasY -> coordenadas dos pontos candidatos
    """

    coordenadas = encontraPontosCandidatos(areaPedunculo)

    todasCoordenadas, coordenadasX, coordenadasY, imagemHUE = coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3]

    pesoY  =  []
    pesoX  =  []
    mediaX =  []
    mediaY =  []

    centro =  areaPedunculo.shape[0] / 2
    centro = int(centro)
     
    #Calcula o peso de Y
    for i in todasCoordenadas:

        aux = calculaPesoY(i)

        pesoY.append(aux)

    #Calcula o peso de X
    for i in todasCoordenadas:

        aux = calculaPesoX(i, centro)

        pesoX.append(aux)

    #Somatorio do peso Y
    somatorioPesoY = sum(pesoY)

    #Somatoria do peso X
    somatorioPesoX = sum(pesoX)

    #Realiza combinação
    combinacaoX = combinacao(pesoX, coordenadasX)
    combinacaoY = combinacao(pesoY, coordenadasY)

    #Calcula a media dos pontos X
    for i in combinacaoX:

        aux = calculaMediaPontos(i)

        mediaX.append(aux)

    #Calcula a media dos pontos Y
    for i in combinacaoY:

        aux = calculaMediaPontos(i)

        mediaY.append(aux)


    somatorioX =  sum(mediaX)
    somatorioY =  sum(mediaY)

    pontoX = (somatorioX / somatorioPesoX)  
    pontoY = (somatorioY / somatorioPesoY)

    pontoX = int(pontoX)
    pontoY = int(pontoY)
    
    cv2.circle(imagemHUE, (pontoX, pontoY), 6, (255,0,0), -1)

    return pontoX, pontoY, imagemHUE

def Kmeans(areaPedunculo, qtdBusca):

    pontosCandidatos = encontraPontosCandidatos(areaPedunculo)

    coordenadas, imagemHUE = pontosCandidatos[0], pontosCandidatos[3]

    pontosKmeans = np.float32(coordenadas)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(pontosKmeans, qtdBusca, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    pontoX, pontoY = int(center[:,0]), int(center[:,1])

    cv2.circle(imagemHUE, (pontoX, pontoY), 6, (0,0,255), -1)

    return pontoX, pontoY, imagemHUE

def encontraCoordenadasMedia(areaPedunculo):

    """
    Função: Encontrar a coordenada (x, y) referente ao ponto de corte na área do pedúnculo utilizando
    a média aritimética dos pontos candidatos

    Parâmetro: 

    coordenadasX, coordenadasY -> coordenadas dos pontos candidatos

    """

    coordenadas = encontraPontosCandidatos(areaPedunculo)

    coordenadasX, coordenadasY, imagemHUE = coordenadas[1], coordenadas[2], coordenadas[3]
    
    mediaSimplesX = np.mean(coordenadasX)
    mediaSimplesY = np.mean(coordenadasY)
    
    mediaNovoX = int(mediaSimplesX)
    mediaNovoY = int(mediaSimplesY)

    cv2.circle(imagemHUE, (mediaNovoX, mediaNovoY), 6, (0,255,0), -1)
    
    return mediaNovoX, mediaNovoY, imagemHUE

def coordenadaPontoFinal(areaPedunculo, baixo, alto, topLeftX, topLeftY, tipoBusca, qtdPontos):

    """
    Função: Encontra o ponto final na imagem original

    Parâmetro:  areaPedunculo -> imagem da região do pedunculo
    		topLeftX, topLeftY -> coordenada da caixa da fruta
            tipoBusca -> Ponderada
                         Media
                         Kmeans
            qtdPontos -> quantos pontos devem ser encontrados no K-means

            @Ponderada: os pontos candidatos são gerados a partir da média ponderada dos valores do HUE
            @Média: os pontos candidatos são gerados a partir da média aritimética dos valores do HUE
            @Kmeans: os pontos candidatos são gerados a partir do K-means dos valores do HUE
    	   
    """

    areaPedunculo = segmentaAreaPedunculo(areaPedunculo, baixo, alto)

    if(tipoBusca == "Kmeans"):

        pontosCandidatos = Kmeans(areaPedunculo, qtdPontos)

    elif(tipoBusca == "Media"):

        pontosCandidatos = encontraCoordenadasMedia(areaPedunculo)

    else:

        pontosCandidatos = encontraCoordenadasPonderada(areaPedunculo)

    pontoFinalX, pontoFinalY, imagemHUE = pontosCandidatos[0], pontosCandidatos[1], pontosCandidatos[2]

    coordenadaFinalX = topLeftX + pontoFinalX
    coordenadaFinalY = topLeftY + pontoFinalY

    return coordenadaFinalX, coordenadaFinalY, areaPedunculo, imagemHUE

def verifica_pixel(img, pontoY, pontoX):

    """
    Função:

    Parâmetro:

    Ajuste:
    """
            
    encontrou = 0
    
    try:

        if np.any(img[int(pontoY), int(pontoX)] == (0,0,0)):

            encontrou = 0
            #print("veri", encontrou)

        else:

            encontrou = 1
            #print("veri", encontrou)

        return encontrou
    
    except IndexError:
        
        encontrou = 1
        
        return encontrou
    

def localizaPonto(img, pontoY, pontoX):

    """
    Função:

    Parâmetro:

    Ajuste:
    """
    
    copiaX, copiaY = pontoX, pontoY
    
    largura = img.shape[1]
    altura  = img.shape[0]
    
    qtdBusca = 0
    
   # print(largura)
    
    #Inicia a busca pra esquerda
    movimento = 1
    
    controle = 0
    
    while(1):
        
        #print("x - y: ", pontoX, pontoY)
        
        coordenada = verifica_pixel(img, round(pontoY), round(pontoX))
                
        try:
    
            if(coordenada == 1):

                pontoY, pontoX = pontoY, pontoX
                
                return int(pontoY), int(pontoX), qtdBusca
            
            else:
                
                if(controle == 0):
                    
                    pontoX = pontoX - movimento
                
                if(pontoX == 1):
                    
                    pontoX = copiaX
                    movimento = - 1
                    controle = 1
                    #print("ok - x = 1")
                    
                if(pontoX == largura - 1):
                    
                    pontoX = copiaX
                    controle = 2
                    #print("ok - x = largura")
                    
                    
                if(controle == 2 and pontoY != altura - 1):
                    
                    pontoX = pontoX + 1
                    pontoY = pontoY + 1
                    controle = 3
                        
                elif(pontoY != altura - 1):
                    
                    pontoX = pontoX - 1
                    pontoY = pontoY + 1
                    
                else:
                    
                    pontoX = copiaX
                    pontoY = copiaY
                                
            qtdBusca = qtdBusca + 1 
            
        except IndexError:
                        
            return int(copiaY), int(copiaX), qtdBusca

def kmeans2Pontos(areaPedunculo, qtdBusca):

    pontosCandidatos = encontraPontosCandidatos(areaPedunculo)

    cores = [(255,255,255), (255,255,0), (255,140,0), (127,255,0), (128,0,0)]

    coordenadas, imagemHUE = pontosCandidatos[0], pontosCandidatos[3]

    pontosGeral = []
    guardaX = []
    guardaY = []

    pontosKmeans = np.float32(coordenadas)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(pontosKmeans, qtdBusca, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    for i in range(qtdBusca):

        pontoX = int(center[:,0][i])
        pontoY = int(center[:,1][i])

        cv2.circle(imagemHUE, (pontoX, pontoY), 6, (255,255,255), -1)

        pontosGeral.append((pontoX, pontoY))
        guardaX.append(pontoX)
        guardaY.append(pontoY)

    for i in range(qtdBusca - 1):

        cv2.line(imagemHUE, (guardaX[i], guardaY[i]), (guardaX[i+1], guardaY[i+1]), (0,0, 255))

    print(guardaX, guardaY)

    return imagemHUE