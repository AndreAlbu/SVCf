import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

def preveAreaPedunculo(posicao, xt, yt, xb, yb, limiarLargura, limiarAltura, alturaCaixa):
    
    """
    Função: Localizar a área do pedúnculo

    Parâmetro: posicao -> posição da caixa do pedúnculo a cada interação
    		   xt, yt, xb, yb -> correspondem as coordenadas da caixa da fruta
               limiarLargura -> determina a quantidade de áreas do pedúnculo a serem geradas
               limiarAltura -> altura da(s) do pedúnculo

    Ajuste: O limiar de altura é ajustado no intervalo de 0 a 1
            A limiar de largura é um número inteiro >= 1
    
    #Tamanho da região de interesse baseada na caixa da fruta
    Lmax = abs(xt - xb)
    Hmax = abs(yt - yb)

    RoiL = limiarLargura * Lmax
    RoiH = limiarAltura  * Hmax

    #Posiciona a área do pedúnculo no centro da caixa da fruta
    tamanhoCaixaPedunculo = Lmax / limiarLargura

    #Posicionamento das caixas
    if(posicao == 0):
        
        x1 = xt
        x2 = x1 + tamanhoCaixaPedunculo
        
    elif(posicao == 1):
        
        x1 = xt + tamanhoCaixaPedunculo
        x2 = x1 + tamanhoCaixaPedunculo
        
    else:
        
        x1 = xt + posicao * tamanhoCaixaPedunculo
        x2 = x1 + tamanhoCaixaPedunculo

    y1 = abs(yt + alturaCaixa) 
    y2 = abs(RoiH - y1)

    #Converte todas as coordenadas para um número inteiro
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    
    return x1, y1, x2, y2 

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

    y1 = abs(yt + alturaCaixa) 
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

	Ajuste: 

	"""
	try:

		histr0 = cv2.calcHist([image], [0], None, [31], [2, 31])

		# Obtém o maior valor do histograma
		ymax = max(histr0)
		xmax, _ = np.where(histr0 == ymax)

		valorHue = int(xmax)

		#print("Valor do Hue: {}".format(valorHue))

		# Quantidade de pixel - posição    
		return valorHue

	except TypeError:

		return 20

def segmentaAreaPedunculo(areaPedunculo, limiarBaixo, limiarAlto):

	"""
	Função: Segmenta a área do pedúnculo

	Parâmetro: areaPedunculo -> imagem da região do pedunculo (RGB)
	       hMIN, sMIN, vMIN -> limiares mínimos do HSV
	       hMAX, sMAX, vMAX -> limiares máximos do HSV

	Ajuste: Os limiares dependem da cultura em análise
	"""

	try:

		hMIN, sMIN, vMIN = limiarBaixo[0], limiarBaixo[1], limiarBaixo[2]
		hMAX, sMAX, vMAX = limiarAlto[0], limiarAlto[1], limiarAlto[2]

		baixo = np.array([hMIN, sMIN, vMIN])
		alto  = np.array([hMAX, sMAX, vMAX])

		img_hsv = cv2.cvtColor(areaPedunculo, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(img_hsv, baixo, alto)

		areaSegmentada = cv2.bitwise_and(areaPedunculo, areaPedunculo, mask = mask)

		return areaSegmentada

	except:

		print("Area do pedunculo sem informacao")
		return areaPedunculo

def encontraPontosCandidatos(areaPedunculo):

	"""
	Função: Encontra os pontos candidatos ao corte

	Parâmetro: areaPedunculo -> imagem da região do pedunculo
	"""

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	centroX = round(largura / 2)
	centroY = round(altura / 2)

	try:

		areaPedunculo = cv2.cvtColor(areaPedunculo, cv2.COLOR_BGR2HSV)

		valorMaximoHUE = histogramaHSV(areaPedunculo)

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
		#print(quantidadePontosEncontrados)

		if(quantidadePontosEncontrados >= 1):

			return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, quantidadePontosEncontrados

		else:

			#print(f"Encontrou correto else {quantidadePontosEncontrados}")       

			todas_coordenadas, todas_coordenadasX, todas_coordenadasY = [(centroX, centroY)], [(centroX)], [(centroY)]
			return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, quantidadePontosEncontrados

	except:

		#print(f"Encontrou com erro 0")

		todas_coordenadas, todas_coordenadasX, todas_coordenadasY = [(centroX, centroY)], [(centroX)], [(centroY)]
		return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, 0



    #cv2.imwrite("segmentada.png", areaPedunculo)

def verificaPonto(areaPedunculo, pontoX, pontoY):

    """
    Função: verifica os pixel em uma posição na imagem

    Parâmetro: 

    Ajuste:
    """

    try:
            
	    verifica = 0

	    if np.any(areaPedunculo[pontoY, pontoX] < 1):

	        verifica = 0

	    else:

	        verifica = 1

	    return verifica

    except:

    	return 1
    
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

def localizaProximo(pontosX, valorMediaX): 
      
     pontosX = np.asarray(pontosX) 

     idx = (np.abs(pontosX - valorMediaX)).argmin() 

     return idx

def encontraCoordenadasPonderada(areaPedunculo):

    """
    Função: Encontrar a coordenada (x, y) referente ao ponto de corte na área do pedúnculo utilizando
    a média ponderada dos pontos candidatos

    Parâmetro: areaPedunculo -> imagem da região do pedunculo
               todasCoordenadas, coordenadasX, coordenadasY -> coordenadas dos pontos candidatos
    """

    coordenadas = encontraPontosCandidatos(areaPedunculo)

    todasCoordenadas, coordenadasX, coordenadasY, imagemHUE, valorMaximoHUE = coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3], coordenadas[4]

    quantidadePontosEncontrados = len(todasCoordenadas)

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
    
    if(verificaPonto(areaPedunculo, pontoX, pontoY)):

        cv2.circle(imagemHUE, (pontoX, pontoY), 1, (255,0,0), -1)

        return pontoX, pontoY, imagemHUE, valorMaximoHUE, quantidadePontosEncontrados

    else:

        index = localizaProximo(coordenadasY, pontoY)

        pontoX, pontoY = coordenadasX[index], coordenadasY[index]

        cv2.circle(imagemHUE, (pontoX, pontoY), 1, (255,0,0), -1)

        return pontoX, pontoY, imagemHUE, valorMaximoHUE, quantidadePontosEncontrados

    return pontoX, pontoY, imagemHUE, valorMaximoHUE, quantidadePontosEncontrados

def encontraCoordenadasMedia(areaPedunculo):

    """
    Função: Encontrar a coordenada (x, y) referente ao ponto de corte na área do pedúnculo utilizando
    a média aritimética dos pontos candidatos

    Parâmetro: 

    coordenadasX, coordenadasY -> coordenadas dos pontos candidatos

    """

    coordenadas = encontraPontosCandidatos(areaPedunculo)

    coordenadasX, coordenadasY, imagemHUE, valorMaximoHUE = coordenadas[1], coordenadas[2], coordenadas[3], coordenadas[4]
    
    mediaSimplesX = np.mean(coordenadasX)
    mediaSimplesY = np.mean(coordenadasY)
    
    pontoX = int(mediaSimplesX)
    pontoY = int(mediaSimplesY)

    if(verificaPonto(areaPedunculo, pontoX, pontoY)):

        cv2.circle(imagemHUE, (pontoX, pontoY), 1, (0,255,0), -1)

        return pontoX, pontoY, imagemHUE, valorMaximoHUE

    else:

        index = localizaProximo(coordenadasY, pontoY)

        pontoX, pontoY = coordenadasX[index], coordenadasY[index]

        cv2.circle(imagemHUE, (pontoX, pontoY), 1, (0,255,0), -1)

        return pontoX, pontoY, imagemHUE, valorMaximoHUE

def outliners(coordenadas):

	limite = 3
	media = np.mean(coordenadas[1])
	desvio = np.std(coordenadas[1])

	listaNova = []
	score = []

	#print(media)
	#print(desvio)

	for i in coordenadas:

		z_score = (i[1] - media) / desvio

		#z_score = abs(z_score)

		#print(z_score)

		score.append(z_score)


		if(z_score >= limite or z_score <= -limite):

			listaNova.append(i)

	#print("Distancias: {}".format(score))

	#print("Coordenadas: {}".format(listaNova))

	return listaNova

def funcIsolationForest(coordenadas):

	try:

		clf = IsolationForest(random_state=0).fit(coordenadas)
		resul = clf.predict(coordenadas)

		resul = clf.predict(coordenadas)

		cordNovo = []

		for i in range(len(resul)):

		  j = coordenadas[i]

		  if(resul[i] == 1):

		    cordNovo.append(j)

		#print("Coordenadas:[y,x] {}".format(cordNovo))

		return cordNovo

	except ValueError:

		#print("Erro IsolationForest")

		return [0]

def pontoCentro(areaPedunculo):

	coordenadas = encontraPontosCandidatos(areaPedunculo)

	imagemHUE, valorMaximoHUE = coordenadas[3], coordenadas[4]

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	centroX = round(largura / 2)
	centroY = round(altura / 2)

	return int(centroX), int(centroY), imagemHUE, valorMaximoHUE

def Kmeans(areaPedunculo, qtdBusca, metodo):

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	qtdPontosAreaPedunculo = largura * altura * 0.7

	try:

		pontosCandidatos = encontraPontosCandidatos(areaPedunculo)

		cores = [(255,255,255), (255,255,0), (255,140,0), (127,255,0), (128,0,0), (50,50,50), (255,255,30)]

		coordenadas, imagemHUE, valorMaximoHUE, qtdPontosEncontrados = pontosCandidatos[0], pontosCandidatos[3], pontosCandidatos[4], pontosCandidatos[5]

		qtdPontosEncon = len(coordenadas)

		if( qtdPontosEncon >= qtdBusca and qtdPontosEncon < qtdPontosAreaPedunculo):

			pontosGeral = []
			guardaX = []
			guardaY = []

			pontosKmeans = np.float32(coordenadas)

			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

			ret, label, center = cv2.kmeans(pontosKmeans, qtdBusca, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

			for i in range(qtdBusca):

			    pontoX = int(center[:,0][i])
			    pontoY = int(center[:,1][i])

			    if(verificaPonto(areaPedunculo, pontoX, pontoY)):

			        cv2.circle(imagemHUE, (pontoX, pontoY), 1, (128,0,0), -1)

			        pontosGeral.append((pontoY, pontoX))
			        guardaX.append(pontoX)
			        guardaY.append(pontoY)

			pontosGeral.sort()

			if(metodo == 2):

				isolation = funcIsolationForest(pontosGeral)

				tamanho = len(isolation)

				if(tamanho > 0):

					posicao = round((tamanho / 2) + 1)

					for i in range(tamanho - 1):

						cv2.line(imagemHUE, (isolation[i][1], isolation[i][0]), (isolation[i+1][1], isolation[i+1][0]), (cores[i]), 2)

					
					pontoX, pontoY = isolation[posicao][1], isolation[posicao][0]

					pontoX, pontoY = int(pontoX), int(pontoY)

				else:

					idx = guardaX.index(np.max(guardaX))

					pontoX, pontoY = isolation[idx], isolation[idx]

					pontoX, pontoY = int(pontoX), int(pontoY)

				cv2.circle(imagemHUE, (pontoX, pontoY), 1, (255,255,255), -1)

			else:

				outLin = outliners(pontosGeral)

				tamanho = len(outLin)

				if(tamanho > 0):

					posicao = round((tamanho / 2) + 1)

					for i in range(tamanho - 1):

						cv2.line(imagemHUE, (outLin[i][1], outLin[i][0]), (outLin[i+1][1], outLin[i+1][0]), (cores[i]), 2)

					
					pontoX, pontoY = outLin[posicao][1], outLin[posicao][0]

					pontoX, pontoY = int(pontoX), int(pontoY)

				else:

					idx = guardaY.index(np.max(guardaY))

					pontoX, pontoY = guardaX[idx], guardaY[idx]

					pontoX, pontoY = int(pontoX), int(pontoY)

				cv2.circle(imagemHUE, (pontoX, pontoY), 1, (255,255,255), -1)

			return pontoX, pontoY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

		else:

			poCentro = pontoCentro(areaPedunculo)

			pontoX, pontoY, imagemHUE, valorMaximoHUE = poCentro[0], poCentro[1], poCentro[2], poCentro[3]

			return pontoX, pontoY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

	except IndexError or TypeError:

		poCentro = pontoCentro(areaPedunculo)

		print("Ponto de corte definido para o centro")

		pontoX, pontoY, imagemHUE, valorMaximoHUE = poCentro[0], poCentro[1], poCentro[2], poCentro[3]

		return pontoX, pontoY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

def funcao_DBSCAN(areaPedunculo, eps, min_samples):

	try:

		pontosCandidatos = encontraPontosCandidatos(areaPedunculo)

		coordenadas, imagemHUE, valorMaximoHUE, qtdPontosEncontrados = pontosCandidatos[0], pontosCandidatos[3], pontosCandidatos[4], pontosCandidatos[5]

		coordenadas = np.array(coordenadas)

		#print(coordenadas)

		dbscan = DBSCAN(eps = eps, min_samples = min_samples)

		dbscan.fit(coordenadas)

		labels = dbscan.labels_

		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

		centers = []

		for i in range(n_clusters):

			#indices = np.where(labels == i)[0]
			#indices = np.array(indices, dtype=int)
			cluster_points = coordenadas[labels == i]
			#cluster_points = coordenadas[indices]
			center = np.mean(cluster_points, axis=0)
			center = center.astype(int)
			cv2.circle(imagemHUE, (center), 1, (128,0,0), -1)
			centers.append(center)

		centers = sorted(centers, key=lambda coordenadas: coordenadas[1])

		#print(centers)

		isolation = funcIsolationForest(centers)

		#print(isolation)

		tamanho = len(isolation)

		if(tamanho > 0):

			posicao = round((tamanho / 2) + 1)

			for i in range(tamanho - 1):

				#print(f"iso {isolation[i][0]} - {isolation[i][1]}")

				cv2.line(imagemHUE, (isolation[i][0], isolation[i][1]), (isolation[i+1][0], isolation[i+1][1]), (255, 0, 0), 2)
			
			pontoX, pontoY = isolation[posicao][0], isolation[posicao][1]

			pontoX, pontoY = int(pontoX), int(pontoY)

		else:

			idx = guardaX.index(np.max(guardaX))

			pontoX, pontoY = isolation[idx], isolation[idx]

			pontoX, pontoY = int(pontoX), int(pontoY)

		cv2.circle(imagemHUE, (pontoX, pontoY), 1, (255,255,255), -1)


		#coordenada_ordenada = sorted(centers, key=lambda coord: coord[1])

		#max_coord = max(coordenada_ordenada, key=lambda coord: coord[1])

		#print(coordenada_ordenada)
		#print(f"Max {max_coord}")
		#print(f"Quantidade DBSCAN: {len(centers)} ")

		return pontoX, pontoY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

	except IndexError or TypeError:

		poCentro = pontoCentro(areaPedunculo)

		print("Ponto de corte definido para o centro")

		pontoX, pontoY, imagemHUE, valorMaximoHUE = poCentro[0], poCentro[1], poCentro[2], poCentro[3]

		return pontoX, pontoY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados


def coordenadaPontoFinal(areaPedunculo, baixo, alto, topLeftX, topLeftY, tipoBusca, qtdPontos, metodo, esp, min_samples):

    """
    Função: Encontra o ponto final na imagem original

    Parâmetro:  areaPedunculo -> imagem da região do pedunculo
    		topLeftX, topLeftY -> coordenada da caixa da fruta
            tipoBusca -> Ponderada
                         Media
                         Kmeans
            qtdPontos -> quantos pontos devem ser encontrados no K-means
            metodo -> 1 (outliners) 2 (IsolationForest)

            @Ponderada: os pontos candidatos são gerados a partir da média ponderada dos valores do HUE
            @Média: os pontos candidatos são gerados a partir da média aritimética dos valores do HUE
            @Kmeans: os pontos candidatos são gerados a partir do K-means dos valores do HUE
    	   
    """

    areaPedunculo = segmentaAreaPedunculo(areaPedunculo, baixo, alto)

    if(tipoBusca == "Kmeans"):

        pontosCandidatos = Kmeans(areaPedunculo, qtdPontos, metodo)

    elif(tipoBusca == "Media"):

        pontosCandidatos = encontraCoordenadasMedia(areaPedunculo)

    elif(tipoBusca == 'DBSCAN'):

    	pontosCandidatos = funcao_DBSCAN(areaPedunculo, esp, min_samples)

    else:

        pontosCandidatos = encontraCoordenadasPonderada(areaPedunculo)

    pontoFinalX, pontoFinalY, imagemHUE, valorMaximoHUE, qtdPontosEncontrados = pontosCandidatos[0], pontosCandidatos[1], pontosCandidatos[2], pontosCandidatos[3], pontosCandidatos[4]

    coordenadaFinalX = topLeftX + pontoFinalX
    coordenadaFinalY = topLeftY + pontoFinalY

    return coordenadaFinalX, coordenadaFinalY, areaPedunculo, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

	
