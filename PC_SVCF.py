"""
    Nome do Módulo: PC_SVCF

    Descrição:
      Este módulo contém funções para encontrar o ponto de corte final em pedúnculos no contexto do Projeto SVCF.

    Autor:
      Hairon Gonçalves
      André Albuquerque

    Data de Criação:
    Criação: 19/02/2024
    Última Modificação: 19/03/2024

    Requisitos:
        Bibliotecas:
           - Opencv
           - Numpy
           - Json
           - Matplotlib
           - Scikit-Learn
           - SciPy
           - kneed
           - Lidar
            
    Licença:
      Licença MIT

    Notas:
      Essa biblioteca faz parte do Projeto SVCF
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import traceback

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN


from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from kneed import KneeLocator

import Lidar as lidar


#####################################################################################################
#																									#
#											FUNCOES GERAIS       									#
#																									#
#####################################################################################################

def preveAreaPedunculo(xt, yt, xb, yb, limiarLargura, limiarAltura, alturaCaixa):
    
    """
    Função: Prever a área do pedúnculo em cima da área da manga

    """

    print("Função: --> Prever area do pedunculo")

    # Tamanho da região de interesse baseada na caixa da fruta
    Lmax = abs(xt - xb)
    Hmax = (yt - yb)

    RoiL = limiarLargura * Lmax
    RoiH = limiarAltura  * Hmax

    # Posiciona a área do pedúnculo no centro da caixa da fruta
    centro = (xb + xt) / 2
    centroCaixa = RoiL / 2

    x1 = centro - centroCaixa
    x2 = centro + centroCaixa
    
    alturaCaixaInferior = Hmax
    alturaCaixaSuperior = alturaCaixaInferior * limiarAltura
        
    # Calcula as novas coordenadas de y    
    y1 = int(yt + (alturaCaixaSuperior - RoiH) + alturaCaixa)
    y2 = int(y1 + RoiH)
    
    if(y2 < 0):
        
        y1 = int(yt + (alturaCaixaSuperior - RoiH))
        y2 = 5

    # Converte todas as coordenadas para números inteiros
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    
    return x1, y1, x2, y2

def histogramaHSV(area_pedunculo, caminhoSalva):

	"""
	Função: Encontrar o maior valor do histograma na componente HUE

	Parâmetro: Imagem do pedúnculo

	Ajuste: O ajuste deve ser dentro do limite da segmentação

	"""
	print("Função: --> Histograma HSV")

	try:

		histograma = cv2.calcHist([area_pedunculo], [0], None, [31], [2, 31])

		# Obtém o maior valor do histograma
		ymax = max(histograma)

		xmax, _ = np.where(histograma == ymax)

		xmax = max(xmax)

		if(xmax > 0):

			valorHue = int(xmax)

		else:

			xmax = 2

			histograma = 2

		valorHue = int(xmax)

		print(f"Valor maximo HUE = {valorHue}")

		#Plotar o histograma
		plt.plot(histograma)
		plt.title("Histograma HSV")
		plt.xlabel("Valor de HUE")
		plt.ylabel("Frequência")

		plt.axvline(x = valorHue, linestyle = '--', color = 'green', label = 'Valor máximo: ' + str(valorHue))

		plt.legend()

		if(not (caminhoSalva is None)):

			#Salvar o histograma como uma imagem
			plt.savefig(caminhoSalva + "c_histograma_HSV.png")

		plt.clf()
 
		return valorHue

	except TypeError:

		traceback.print_exc()

		print("Erro na função: Valor = 2")

		return 2

def segmentaAreaPedunculo(areaPedunculo, limiarBaixo, limiarAlto):

	"""
	Função: Segmenta a área do pedúnculo

	Parâmetro: areaPedunculo -> 
		   imagem da região do pedúnculo (RGB)
	       hMIN, sMIN, vMIN -> limiares mínimos do HSV
	       hMAX, sMAX, vMAX -> limiares máximos do HSV

	Ajuste: Os limiares dependem da cultura em análise

	Retorno: Imagem em RGB segmentada de acordo com os parêmetros
	"""
	print("Função: --> Segmenta Área Pedúnculo")

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

		print("Erro na função: Segmenta Area Pedunculo")
		return areaPedunculo

def encontraPontosCandidatos(areaPedunculo, caminhoSalva):

	"""
	Função: Encontra os pontos candidatos ao corte

	Parâmetro: areaPedunculo -> imagem da região do pedúnculo

	"""

	print("Função: --> Encontra Pontos Candidatos")

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	centroX = round(largura / 2)
	centroY = round(altura / 2)

	possibilidade_correta = False

	quantidadePontosEncontrados = 0

	try:

		areaPedunculo = cv2.cvtColor(areaPedunculo, cv2.COLOR_BGR2HSV) #Converte a imagem para HSV

		valorMaximoHUE = histogramaHSV(areaPedunculo, caminhoSalva)

		todas_coordenadas  =  [ ] #Guarda todas as coordenadas (eixo X e Y)
		todas_coordenadasX =  [ ] #Guarda as coordenadas do eixo X vindas dos cantos encontrados
		todas_coordenadasY =  [ ] #Guarda as coordenadas do eixo Y vindas dos cantos encontrados

		for x in range(0, largura):

			for y in range(0, altura):

				h, s, v = areaPedunculo[y, x]

				if(h == valorMaximoHUE):

					todas_coordenadas.append((x,y))
					todas_coordenadasX.append(int(x))
					todas_coordenadasY.append(int(y))

					cv2.circle(areaPedunculo, (x, y), 1, (255, 0, 0), -1)

		if(not (caminhoSalva is None)):

			cv2.imwrite(caminhoSalva + "d_pontos_candidatos.jpg", areaPedunculo)

		quantidadePontosEncontrados = len(todas_coordenadas)

		print(f"Quantidade de Pontos Candidatos: {quantidadePontosEncontrados}")

		if(quantidadePontosEncontrados >= 1):

			possibilidade_correta = True

			return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, quantidadePontosEncontrados, possibilidade_correta

		else:

			todas_coordenadas, todas_coordenadasX, todas_coordenadasY = [(centroX, centroY)], [(centroX)], [(centroY)]

			return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, quantidadePontosEncontrados, possibilidade_correta

	except:

		print(f"Erro na função: Encontra Pontos Candidatos")

		todas_coordenadas, todas_coordenadasX, todas_coordenadasY = [(centroX, centroY)], [(centroX)], [(centroY)]

		return todas_coordenadas, todas_coordenadasX, todas_coordenadasY, areaPedunculo, valorMaximoHUE, quantidadePontosEncontrados, possibilidade_correta

def funcao_coordenada_central(areaPedunculo):

	coordenadas = [ ]

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	centroX = round(largura / 2)
	centroY = round(altura / 2)

	centroX = int(centroX)
	centroY = int(centroY)

	coordenadas.append((centroX, centroY))

	return coordenadas

def gerar_cor_contraste():

    while True:

        color = tuple(np.random.randint(0, 255, 3))
        
        if (color[1] > 50 and color[1] > color[0] and color[1] > color[2] and (abs(color[0] - color[1]) > 30 or abs(color[0] - color[2]) > 30)):

            return color
#####################################################################################################
#																									#
#								 FUNCOES PARA UTILIZAR HEURISTICA  									#
#																									#
#####################################################################################################

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

def funcao_heuristica(areaPedunculo, caminhoSalva):

	print("Função --> Função Heuristica")

	coordenadas = encontraPontosCandidatos(areaPedunculo, caminhoSalva)

	todasCoordenadas, coordenadasX, coordenadasY, imagemHUE, valorMaximoHUE = coordenadas[0], coordenadas[1], coordenadas[2], coordenadas[3], coordenadas[4]

	coordenadas_retorno = [ ]

	try:

		quantidadePontosEncontrados = len(todasCoordenadas)

		pesoY  =  [ ]
		pesoX  =  [ ]
		mediaX =  [ ]
		mediaY =  [ ]

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

		cv2.circle(imagemHUE, (pontoX, pontoY), 3, (0, 255, 255), -1)

		coordenadas_retorno.append((pontoX, pontoY))

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, quantidadePontosEncontrados

	except:

		print("Erro desconhecido -> Função Heuristica")

		coodenadas_erro = funcao_coordenada_central(areaPedunculo)

		coordenadas_retorno = coodenadas_erro

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

#####################################################################################################
#																									#
#							      FUNCOES PARA UTILIZAR O KMEANS    								#
#																									#
#####################################################################################################

def funcao_kmeans(areaPedunculo, qtdBusca, caminhoSalva):

	print("Função: --> Método Kmeans")

	largura = areaPedunculo.shape[1]
	altura  = areaPedunculo.shape[0]

	coordenadas_retorno = [ ]

	quantidade_clusters_selecionados = 0

	try:

		pontosCandidatos = encontraPontosCandidatos(areaPedunculo, caminhoSalva)

		coordenadas, imagemHUE, valorMaximoHUE, qtdPontosEncontrados = pontosCandidatos[0], pontosCandidatos[3], pontosCandidatos[4], pontosCandidatos[5]

		qtdPontosEncon = len(coordenadas)

		if(qtdPontosEncon >= qtdBusca):

			pontosGeral = [ ]
			guardaX     = [ ]
			guardaY     = [ ]

			pontosKmeans = np.float32(coordenadas)

			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

			ret, label, center = cv2.kmeans(pontosKmeans, qtdBusca, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

			for i in range(qtdBusca):

				color = gerar_cor_contraste()

				color = (int(color[0]), int(color[1]), int(color[2]))

				coordenadas_np = np.array(coordenadas)

				for j in range(len(coordenadas)):

					if label[j] == i:

						ponto = coordenadas_np[j].astype(int)
						pontoX_cluster = ponto[0]
						pontoY_cluster = ponto[1]

						cv2.circle(imagemHUE, (pontoX_cluster, pontoY_cluster), 1, color, -1)

				pontoX = int(center[:,0][i])
				pontoY = int(center[:,1][i])

				cv2.circle(imagemHUE, (pontoX, pontoY), 3, color, -1)
				cv2.circle(imagemHUE, (pontoX, pontoY), 4, (255, 255, 255), 1)        

				pontosGeral.append((pontoX, pontoY))
				guardaX.append(pontoX)
				guardaY.append(pontoY)

			coordenadas_retorno = pontosGeral

		else:

			print("A quantidade de pontos candidatos é menor que o limite de cluster")

			coodenadas_erro = funcao_coordenada_central(areaPedunculo)

			coordenadas_retorno = coodenadas_erro

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

	except:

		traceback.print_exc()

		print("Erro desconhecido: Função --> Método Kmeans")

		coodenadas_erro = funcao_coordenada_central(areaPedunculo)

		coordenadas_retorno = coodenadas_erro

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, qtdPontosEncontrados, quantidade_clusters_selecionados

#####################################################################################################
#																									#
#							      FUNCOES PARA UTILIZAR O DBSCAN    								#
#																									#
#####################################################################################################

def localizaEps(k, coordenadas, caminhoSalva):
    
    if(len(coordenadas) <= k):
        
        return k

    try:

        nbrs = NearestNeighbors(n_neighbors = k).fit(coordenadas)

        distancias, indices = nbrs.kneighbors(coordenadas)
       
        distancias = np.sort(distancias[:, k-1], axis = 0)

        kl = KneeLocator(range(len(distancias)), distancias, curve = 'convex')

        if(kl.elbow is None or kl.elbow < 1):

        	print("Erro: Curva não encontrada")

        	eps = k

        else:

        	epsilon = int(kl.elbow)

	        eps = int(distancias[epsilon])

	        plt.title("Vizinhos Próximos")

	        plt.plot(distancias)

	        plt.axvline(x = epsilon, linestyle = '--', color = 'red')
	        plt.axhline(y = eps,     linestyle = '--', color = 'green')

	       	plt.xlabel("Quantidade de Pontos Ordenados")
	       	plt.ylabel("Distância do 4° Vizinho Mais Próximo")

	        plt.text(epsilon, eps-2.5, f'eps = {eps}')

	        if(not (caminhoSalva is None)):

	        	plt.savefig(caminhoSalva + "_curva_eps.png")

	        plt.clf()

        return eps

    except UserWarning:

    	print("Erro desconhecido: Função --> localizaEps")

    	eps = k

    	return eps

def funcao_dbscan(areaPedunculo, min_samples, caminhoSalva):

	print("Função: --> Método DBSCAN")

	coordenadas_retorno = [ ]

	try:

		pontosCandidatos = encontraPontosCandidatos(areaPedunculo, caminhoSalva)

		coordenadas, imagemHUE, valorMaximoHUE, qtdPontosEncontrados = pontosCandidatos[0], pontosCandidatos[3], pontosCandidatos[4], pontosCandidatos[5]

		coordenadas = np.array(coordenadas)

		eps = localizaEps(min_samples, coordenadas, caminhoSalva)

		dbscan = DBSCAN(eps = eps, min_samples = min_samples)

		dbscan.fit(coordenadas)

		labels = dbscan.labels_

		numero_clusters = len(set(labels)) - (1 if -1 in labels else 0)

		print(f"Quantidade de cluster encontrado: {numero_clusters}")

		if(numero_clusters > 0):

			densidade = []
			centroids = []

			for i in range(numero_clusters):

				pontos_clusters = coordenadas[labels == i]

				pontos_clusters = pontos_clusters.astype(int)

				color = gerar_cor_contraste()

				color = (int(color[0]), int(color[1]), int(color[2]))

				for j in range(len(pontos_clusters)):

					cv2.circle(imagemHUE, (pontos_clusters[j][0], pontos_clusters[j][1]), 1, color, -1)

				centroi = np.mean(pontos_clusters, axis = 0).astype(int)

				centroids.append((centroi))

				cv2.circle(imagemHUE, (centroi), 3, color, -1)
				cv2.circle(imagemHUE, (centroi), 4, (255, 255, 255), 1)

			coordenadas_retorno = centroids

		else:

			print("Não encontrou nenhum cluster")

			coodenadas_erro = funcao_coordenada_central(areaPedunculo)

			coordenadas_retorno = coodenadas_erro

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

	except:

		print("Erro desconhecido: Função --> Método DBSCAN")

		traceback.print_exc()

		coodenadas_erro = funcao_coordenada_central(areaPedunculo)

		coordenadas_retorno = coodenadas_erro

		return coordenadas_retorno, imagemHUE, valorMaximoHUE, qtdPontosEncontrados

#####################################################################################################
#																									#
#							      FUNCOES PARA REMOVER OUTLINERS    								#
#																									#
#####################################################################################################

def remove_pontos_fc_outliners(coordenadas, limite):

	print("Função: --> Método Outliners")

	valores = [coord[1] for coord in coordenadas]

	media = np.mean(valores)
	desvio = np.std(valores)

	listaNova = []

	for coord in coordenadas:

		z_score = (coord[1] - media) / desvio

		if abs(z_score) <= limite:

			listaNova.append(coord)

	return listaNova

def remove_pontos_fc_isolation_forest(coordenadas, distancias):

	print("Função: --> Método IsolationForest")

	try:

		distancias_filtradas = 0

		coordenadas = np.array(coordenadas)

		clf = IsolationForest(random_state=0).fit(coordenadas)

		resul = clf.predict(coordenadas)

		indices_validos = resul == 1

		coordenadas_filtradas = coordenadas[indices_validos]

		if(not (distancias is None)):

			distancias = np.array(distancias)

			distancias_filtradas = distancias[indices_validos]

		return coordenadas_filtradas, distancias_filtradas

	except ValueError:

		print("Erro na função: Isolation Forest  -------------->>Erro<<--------------")

		return [0]

def remove_pontos_dc(coordenadas, imagemHUE, metodo, altura_coordenadas):

	print("Função: --> Remove Pontos Fora da Curva")

	coordenadas_pontos_dc = [ ]

	if(metodo == "IsolationForest"):

		f_isolation = remove_pontos_fc_isolation_forest(coordenadas, altura_coordenadas)

		if(len(f_isolation) > 1):

			isolation, distancias_correta = f_isolation[0], f_isolation[1]

			tamanho_isolation = len(isolation)

			print(f"Quantidade de pontos selecionados: {tamanho_isolation}")

			quantidade_clusters_selecionados = tamanho_isolation

			if(tamanho_isolation > 0):

				for i_isolation in range(tamanho_isolation):
					
					cv2.circle(imagemHUE, (isolation[i_isolation][0], isolation[i_isolation][1]), 5, (0, 255, 255), 2)

				coordenadas_pontos_dc = isolation

			else:

				print("IsolationForest = ------------------------------------>>>0<<<------------------------------------")

				coodenadas_erro = funcao_coordenada_central(imagemHUE)

				coordenadas_pontos_dc = coodenadas_erro

		else:

			print("IsolationForest = ------------------------------------>>>Nao clusterizado<<<------------------------------------")

			coodenadas_erro = funcao_coordenada_central(imagemHUE)

			coordenadas_pontos_dc = coodenadas_erro

	elif(metodo == "Outliners"):

		limite = 1

		outLin = remove_pontos_fc_outliners(coordenadas, limite)

		tamanho_outliners = len(outLin)

		print(f"Quantidade de pontos selecionados: {tamanho_outliners}")

		coordenadas_pontos_dc = tamanho_outliners

		if(tamanho_outliners > 0):

			for i_outliners in range(tamanho_outliners):

				cv2.circle(imagemHUE, (outLin[i_outliners][0], outLin[i_outliners][1]), 5, (0, 255, 255), 2)

			coordenadas_pontos_dc = outLin

		else:

			print("Outliners = ------------------------------------>>>0<<<------------------------------------")

			coodenadas_erro = funcao_coordenada_central(imagemHUE)

			coordenadas_pontos_dc = coodenadas_erro

	return coordenadas_pontos_dc, imagemHUE, distancias_correta

#####################################################################################################
#																									#
#							    FUNCOES PARA SELECIONAR PONTO FINAL 3D 								#
#																									#
#####################################################################################################

def obter_cm_px(numero_raios, tamanho, x, y, pontosJson):

	print("Função: --> Obter CM/PX")

	increm =  1
	raioFinal = 0

	somaCM = 0
	somaPX = 0

	while(increm <= numero_raios):

		raioFinal = increm * tamanho

		x_1 = x - raioFinal
		x1  = x + raioFinal

		distancias = lidar.measureDistance((x_1, y), (x1, y), pontosJson)

		#cv2.circle(area_manga, (int(x), int(y)), int(raioFinal), (0, 0, 255))

		#cv2.circle(area_manga, (int(x_1),int(y)), 2, (0, 255, 255))

		#cv2.circle(area_manga, (int(x1), int(y)), 2, (0, 255, 255))

		somaCM += distancias
		somaPX += (raioFinal * 2)

		increm += 1

	cm_px = (somaCM / somaPX) * 100

	return cm_px

def calcula_distancias_entre_pontos(ponto1, ponto2, pontosJson, cm_px):

	distancia_p1_p2 = (((ponto1[0] - ponto2[0]) ** 2) + ((ponto1[1] - ponto2[1]) ** 2)) ** 0.5

	distancia_final = distancia_p1_p2 * cm_px

	return distancia_final

def seleciona_ponto_altura_correta(ponto1, altura, cm_px, fator_imgs_3D):

	print("O ponto foi definido para o centro")

	ponto2_y = int((ponto1[1] - (fator_imgs_3D * altura / cm_px)))

	ponto2_x = ponto1[0]

	ponto2 = (ponto2_x, ponto2_y)

	return ponto2

def seleciona_candidatos_3D(area_pedunculo, coordenadasCandidatas, pontosJson, cm_px, fator_imgs_3D, pontoCentroTopoCaixa, TopLeftX, TopLeftY, altura_minima_ponto, metodo):
    
    qtdPontoCorretosCorte = 0

    largura_x = area_pedunculo.shape[1]

    print("Função: --> Seleciona pontos 3D")

    coordenadas_proximas     = [ ]
    distancias_pontos_cortes = [ ]
    distancias_gerais = [ ]
    
    quantidadePontosEncontrados = len(coordenadasCandidatas)
    
    area_pedunculo_copy = np.copy(area_pedunculo)

    x_centro_caixa = int(pontoCentroTopoCaixa[0] / fator_imgs_3D)
    y_centro_caixa = int(pontoCentroTopoCaixa[1] / fator_imgs_3D)
    
    if(quantidadePontosEncontrados >= 1):
                
        for coord in coordenadasCandidatas:
            
            x_lidar = int((coord[0] + TopLeftX) / fator_imgs_3D)
            y_lidar = int((coord[1] + TopLeftY) / fator_imgs_3D)
                                                           
            distanciaPontoCandidato = calcula_distancias_entre_pontos((x_centro_caixa, y_centro_caixa), (x_lidar, y_lidar), pontosJson, cm_px)

            distancias_gerais.append(distanciaPontoCandidato)
            
            if(distanciaPontoCandidato >= altura_minima_ponto):
                
                #print(f"PCan <--> TC {distanciaPontoCandidato}")
                
                qtdPontoCorretosCorte += 1
                
                coordenadas_proximas.append((coord))
                
                distancias_pontos_cortes.append(distanciaPontoCandidato)
                
                cv2.circle(area_pedunculo_copy, (coord), 5, (255, 0, 0), 2)
                
                #print(f"Pontos corretos: ({coord})")

        print(f"Quantidade de pontos para corte: {qtdPontoCorretosCorte}")
            
    else:
            
        print("Não existe pontos candidatos")

    if(qtdPontoCorretosCorte < 1):   

    	print("Ponto no centro do area do pedúnculo") 		

    	altura_correta = altura_minima_ponto / cm_px * fator_imgs_3D

    	x_erro, y_erro = pontoCentroTopoCaixa[0], pontoCentroTopoCaixa[1]

    	x_erro, y_erro = int(x_erro - TopLeftX), int(y_erro - TopLeftY)

    	if(len(coordenadasCandidatas) > 0):

    		if(metodo == "Heuristica"):

    			x_erro = int(coordenadasCandidatas[0][0])

    		else:

    			pdc = 3
    			pd3 = 2

    			lista_pesos = [ ]

    			largura_x = int(largura_x / 2)

    			print(coordenadasCandidatas)
    			print(distancias_gerais)

    			for p in range(len(coordenadasCandidatas)):

    				dc = abs(coordenadasCandidatas[p][0] - largura_x)

    				d3 = abs(distancias_gerais[p] - 3)

    				dpa = (dc * pdc + d3 * pd3) / (pdc + pd3)

    				lista_pesos.append(dpa)

    			index = lista_pesos.index(min(lista_pesos))

    			x_erro = int(coordenadasCandidatas[index][0])

    	coord_altura_correta = (x_erro, int(y_erro - altura_correta))

    	print(coord_altura_correta)

    	coordenadas_proximas.append((coord_altura_correta))

    	distancias_pontos_cortes.append(altura_minima_ponto)

    return coordenadas_proximas, distancias_pontos_cortes, area_pedunculo_copy


#####################################################################################################
#																									#
#							    FUNCOES PARA SELECIONAR PONTO FINAL 								#
#																									#
#####################################################################################################

def seleciona_ponto(coordenadas, imagemHUE, distancias_correta, tipoBase, largura_caixa_pedunculo, altura_manga_px, pontosJson, topLeftX, topLeftY):

	print("Função: --> Seleciona ponto final")

	distancia_ponto_final = -1

	largura_x = int(largura_caixa_pedunculo / 2)

	lista_pesos = [ ]

	pdc = 3
	pd3 = 2

	if(tipoBase == "2D"):

		for p in range(len(coordenadas)):

			dc = abs(coordenadas[p][0] - largura_x)

			dm = abs(coordenadas[p][1] - altura_manga_px)

			dpa = (dc * pdc + dm * pd3) / (pdc + pd3)

			lista_pesos.append(dpa)

		index = lista_pesos.index(min(lista_pesos))

		cv2.circle(imagemHUE, coordenadas[index], 5, (0, 0, 255), -2)

		print(f"Coordenada selecionada: {coordenadas[index]}")

		coord_final = coordenadas[index]

	else:

		for p in range(len(coordenadas)):

			dc = abs(coordenadas[p][0] - largura_x)

			d3 = abs(distancias_correta[p] - 3)

			dpa = (dc * pdc + d3 * pd3) / (pdc + pd3)

			lista_pesos.append(dpa)

		index = lista_pesos.index(min(lista_pesos))

		cv2.circle(imagemHUE, coordenadas[index], 5, (0, 0, 255), -2)

		distancia_ponto_final = distancias_correta[index]

		distancia_ponto_final = round(distancia_ponto_final, 2)

		print(f"A distância do ponto selecionado para caixa IA: {distancia_ponto_final} cm")

		coord_final = coordenadas[index]

		x_distancia = int((coord_final[0] + topLeftX) / 7.5)
		y_distancia = int((coord_final[1] + topLeftY) / 7.5)

		#distanciaProfundidadePonto = lidar.measureDistanceOnePoint((x_distancia, y_distancia), pontosJson)

		#print(f"Distancia profundidade ponto selecionado: {round(distanciaProfundidadePonto, 2)} cm")

	return coord_final, imagemHUE, distancia_ponto_final

#####################################################################################################
#																									#
#							                    FUNCAO PRINCIPAL    								#
#																									#
#####################################################################################################

def localiza_ponto_final(id_imagem, id_manga_localizada, areaPedunculo, baixo, alto, metodoCluster, caminhoSalva, removePontos, posicaoDesejada, quantidadeBusca, min_samples, tipoBase, metodoFC, topLeftX, topLeftY, pontosJson, alturaMinima, centroAreaManga, pontoCentroTopoCaixa):

	#Primeiro -> segmenta a área do pedúnculo
	#Segundo  -> faz o histograma para encontrar o máximo do hue
    #Terceiro -> encontrar os pontos candidatos
    #Quarto   -> clusterizar os pontos candidatos
    #Quinto   -> remover pontos clusterizados fora da curva

    areaPedunculo = segmentaAreaPedunculo(areaPedunculo, baixo, alto)

    fator_imgs_3D = 7.5
    distancia_ponto_final = -1

    largura_caixa_pedunculo = areaPedunculo.shape[1]

    y_manga_pixel = areaPedunculo.shape[0]

    caminhoSalva = caminhoSalva + id_imagem + "_" + id_manga_localizada + "_"

    if(metodoCluster == "Kmeans"):

        resultado_clusterizacao = funcao_kmeans(areaPedunculo, quantidadeBusca, caminhoSalva)

    elif(metodoCluster == 'DBSCAN'):

    	resultado_clusterizacao = funcao_dbscan(areaPedunculo, min_samples, caminhoSalva)

    else:

    	resultado_clusterizacao = funcao_heuristica(areaPedunculo, caminhoSalva)

    coordenadas_clusterizadas, imagemHUE, valorHue, qtdPontosEncontrados = resultado_clusterizacao[0], resultado_clusterizacao[1], resultado_clusterizacao[2], resultado_clusterizacao[3]

    imagemHUE_ = imagemHUE.copy()

    if(tipoBase == "2D"):

    	if(removePontos and metodoCluster != "Heuristica"):

	    	pontos_dentro_curva = remove_pontos_dc(coordenadas_clusterizadas, imagemHUE, metodoFC, None)

	    	coordenadas_clusterizadas, imagemHueDC = pontos_dentro_curva[0], pontos_dentro_curva[1]

    		imagemHueDC_ = imagemHueDC.copy()

    	info_2D = seleciona_ponto(coordenadas_clusterizadas, imagemHUE, None, tipoBase, largura_caixa_pedunculo, y_manga_pixel, None, None, None)

    	pontoFinalX, pontoFinalY, imagemHueD = int(info_2D[0][0]), int(info_2D[0][1]), info_2D[1]

    	imagemHueD_ = imagemHueD.copy()

    elif(tipoBase == "3D"):

    	centroMangaX, centroMangaY = centroAreaManga[0], centroAreaManga[1]

    	centroMangaX, centroMangaY = int(centroMangaX / fator_imgs_3D), int(centroMangaY / fator_imgs_3D)

    	fator_cm_px = obter_cm_px(4, 2, centroMangaX, centroMangaY, pontosJson)

    	candidatos_corretos = seleciona_candidatos_3D(imagemHUE_, coordenadas_clusterizadas, pontosJson, fator_cm_px, fator_imgs_3D, pontoCentroTopoCaixa, topLeftX, topLeftY, alturaMinima, metodoCluster)

    	coordenadas_corretas, distancias_correta, imagemHueCorreta = candidatos_corretos[0], candidatos_corretos[1], candidatos_corretos[2]

    	imagemHueCorreta_ = imagemHueCorreta.copy()

    	distancias_corretas_dc = distancias_correta

    	imagemHueDC = imagemHueCorreta.copy()

    	if(removePontos and metodoCluster != "Heuristica"):

	    	pontos_dentro_curva = remove_pontos_dc(coordenadas_corretas, imagemHUE, metodoFC, distancias_correta)

	    	coordenadas_corretas, imagemHueDC, distancias_corretas_dc = pontos_dentro_curva[0], pontos_dentro_curva[1], pontos_dentro_curva[2]

    		imagemHueDC_ = imagemHueDC.copy()

    	info_3D = seleciona_ponto(coordenadas_corretas, imagemHueDC, distancias_corretas_dc, tipoBase, largura_caixa_pedunculo, None, pontosJson, topLeftX, topLeftY)

    	pontoFinalX, pontoFinalY, imagemHueD, distancia_ponto_final = info_3D[0][0], info_3D[0][1], info_3D[1], info_3D[2]

    	imagemHueD_ = imagemHueD.copy()


    if(not (caminhoSalva is None)):
            
    	cv2.imwrite(caminhoSalva + "e_clusterizacao.jpg", imagemHUE_)
    	cv2.imwrite(caminhoSalva + "h_ponto_final.jpg", imagemHueD_)

    	if(removePontos and metodoCluster != "Heuristica"):

    		cv2.imwrite(caminhoSalva + "g_dentro_curva.jpg", imagemHueDC_)

    	if(tipoBase == "3D"):

    		cv2.imwrite(caminhoSalva + "f_verificacao_distancias.jpg", imagemHueCorreta)

    coordenadaFinalX = topLeftX + pontoFinalX
    coordenadaFinalY = topLeftY + pontoFinalY
    
    return coordenadaFinalX, coordenadaFinalY, valorHue, distancia_ponto_final
