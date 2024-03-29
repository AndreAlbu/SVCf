"""
    Nome do Módulo: PC_SVCF

    Descrição:
      Este módulo contém funções para validar as imagens marcadas no contexto do Projeto SVCF.

    Autor:
      Hairon Gonçalves
      André Albuquerque

    Data de Criação:
    Criação: 19/02/2024
    Última Modificação: 25/03/2024

    Requisitos:
        Bibliotecas:
           - Opencv
           - Numpy
           - Json
           - Lidar
           - PC_SVCF
            
    Licença:
      Licença MIT

    Notas:
      Essa biblioteca faz parte do Projeto SVCF
"""

import json
import numpy as np
import cv2

import Lidar as lidar
import PC_SVCF as pc

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR ÁREA DA MANGA    	     					#
#																									#
#####################################################################################################

def ler_coordenadas_conhecidas(id_imagem, id_manga_localizada, ARQUIVO_JSON, TIPO_BASE):

	with open(ARQUIVO_JSON) as file:

		coordenadas = json.load(file)

	informacao_imagem = coordenadas.get('InformationImage')

	if informacao_imagem is None:

		print(f"{id_imagem}: não existe informações de posição das caixas")

		print("<-------------------------------------->")

		return

	REDUCAO_IMAGEM   = coordenadas['InformationImage']['reduction_value']

	MANGA_INCLINADA = False
	inclinado = 0

	if(TIPO_BASE == "2D"):

		MANGA_INCLINADA  = coordenadas['InformationImage']['mango_inclined']

	REDUCAO_IMAGEM = 1

	coordenadas_manga     = [ ]
	coordenadas_pedunculo = [ ]
	coordenadas_ponto     = [ ]

	#Coordenadas da manga
	xt, yt = coordenadas['MangoCoordinates_' + str(id_manga_localizada)]['xt'], coordenadas['MangoCoordinates_' + str(id_manga_localizada)]['yt']
	xb, yb = coordenadas['MangoCoordinates_' + str(id_manga_localizada)]['xb'], coordenadas['MangoCoordinates_' + str(id_manga_localizada)]['yb']
	coordenadas_manga.append((xt, yt, xb, yb))

	#Coordenadas do pedúnculo
	if(MANGA_INCLINADA):

		inclinado = coordenadas['PeduncleInclined_' + str(id_manga_localizada)]['value']

		inclinado = inclinado + 1

		for inclinacao in range(1, inclinado):

			xt, yt = coordenadas['PeduncleCoordinates_' + str(id_manga_localizada) + "_" + str(inclinacao)]['xt'] / REDUCAO_IMAGEM, coordenadas['PeduncleCoordinates_' + str(id_manga_localizada) + "_" + str(inclinacao)]['yt'] / REDUCAO_IMAGEM
			xb, yb = coordenadas['PeduncleCoordinates_' + str(id_manga_localizada) + "_" + str(inclinacao)]['xb'] / REDUCAO_IMAGEM, coordenadas['PeduncleCoordinates_' + str(id_manga_localizada) + "_" + str(inclinacao)]['yb'] / REDUCAO_IMAGEM
			coordenadas_pedunculo.append((int(xt), int(yt), int(xb), int(yb)))

	else:

		xt, yt = coordenadas['PeduncleCoordinates_' + str(id_manga_localizada)]['xt'], coordenadas['PeduncleCoordinates_' + str(id_manga_localizada)]['yt']
		xb, yb = coordenadas['PeduncleCoordinates_' + str(id_manga_localizada)]['xb'], coordenadas['PeduncleCoordinates_' + str(id_manga_localizada)]['yb']
		coordenadas_pedunculo.append((xt, yt, xb, yb))

		#Coordenadas do ponto
		if(TIPO_BASE == "3D"):

			xt, yt = coordenadas['PointCoordinates_' + str(id_manga_localizada)]['xt'], coordenadas['PointCoordinates_' + str(id_manga_localizada)]['yt']
			coordenadas_ponto.append((xt, yt))

	return coordenadas_manga, coordenadas_pedunculo, coordenadas_ponto, inclinado

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR ÁREA DA MANGA    	     					#
#																									#
#####################################################################################################

def calcula_area_retangulo(coordenadas_area):
    
    return (coordenadas_area[2] - coordenadas_area[0]) * (coordenadas_area[3] - coordenadas_area[1])

def calcula_intersecao(coord1, coord2):
    
    x1 = max(coord1[0], coord2[0])
    y1 = max(coord1[1], coord2[1])
    x2 = min(coord1[2], coord2[2])
    y2 = min(coord1[3], coord2[3])

    if x1 < x2 and y1 < y2:
        
        return (x1, y1, x2, y2)
    
    else:
        
        return None
    
def porcentagem_intersecao_area_manga(coord1, coord2):
    
    calcula_intersecao_areas = calcula_intersecao(coord1, coord2)
    
    intersecao_areas = calcula_area_retangulo(calcula_intersecao_areas) if calcula_intersecao_areas else 0

    area_total = calcula_area_retangulo(coord1)
    
    porcentagem = (intersecao_areas / area_total) * 100
    
    return porcentagem

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR ÁREA DO PEDÚNCULO   	     					#
#																									#
#####################################################################################################

def area_pedunculo(imagem, pontoX, pontoY, coordenadas_pedunculo, inclinacao, cm_px_2D):

	encontrou = False

	for coordenadas_atual in coordenadas_pedunculo:

		xtP, ytP, xbP, ybP = coordenadas_atual

		cv2.rectangle(imagem, (int(xtP - cm_px_2D), int(ytP)), (int(xbP + cm_px_2D), int(ybP)), (0, 255, 255,), 1)

		if((pontoX >= xtP - cm_px_2D) and (pontoX <= xbP + cm_px_2D) and (pontoY >= ytP - cm_px_2D) and (pontoY <= ybP + cm_px_2D)):

			encontrou = True
			break

	return encontrou

def calcular_intersecao_porcentagem(coord1, coord2):
    
    tamanho_caixa_ia = coord1[2] - coord1[0]
    
    inicio_x = max(coord1[0], coord2[0])
    fim_x = min(coord1[2], coord2[2])
    
    if fim_x >= inicio_x:

        intersecao_x = fim_x - inicio_x
        
        if coord2[0] >= coord1[0] and coord2[2] <= coord1[2]:

            porcentagem_intersecao_area_pedunculo = (intersecao_x / (coord2[2] - coord2[0])) * 100

        else:

            porcentagem_intersecao_area_pedunculo = (intersecao_x / tamanho_caixa_ia) * 100

        return porcentagem_intersecao_area_pedunculo

    else:

        return 0

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR PONTO DE CORTE    	     					#
#																									#
#####################################################################################################

def encontra_area_px_ponto_corte(coord, cm_px, fator_imgs):
        
    raio = np.sqrt(coord / np.pi)
    
    area_px = raio / cm_px * fator_imgs

    return round(area_px)

#####################################################################################################
#																									#
#							                    FUNCAO PRINCIPAL    								#
#																									#
#####################################################################################################

def valida_svcf(imagem, caminhoSalva, id_imagem, id_manga_localizada, tipo_base, coordenada_manga_ia, coordenada_pedunculo, arquivo_json, fator_cm_px, coordenadas_ponto, distancia_minima, distancia_ponto, cm_px_2D, fator_ponto_area):

	fator_imgs = 7.5

	distancia_horizontal    = 0
	distanciaPontoConhecido = 0
	distancia_vertical      = 0
	distanciaCaixasManga    = 0

	porcentagem_pedunculo = -1

	#Coordenadas da área do pedúnculo
	pedunculoX, pedunculoY = coordenada_pedunculo[0], coordenada_pedunculo[2]

	#Coordenadas do ponto de corte final
	pontoX, pontoY = coordenadas_ponto[0], coordenadas_ponto[1]

	#Função para ler o Json com as coordenadas conhecidas
	coordenadas_conhecidas = ler_coordenadas_conhecidas(id_imagem, id_manga_localizada, arquivo_json, tipo_base)

	#Porcentagem Manga
	xtM, ytM, xbM, ybM = coordenadas_conhecidas[0][0][0], coordenadas_conhecidas[0][0][1], coordenadas_conhecidas[0][0][2], coordenadas_conhecidas[0][0][3]
	
	porcentagem_manga = 0.0

	porcentagem_manga = porcentagem_intersecao_area_manga(coordenada_manga_ia, (xtM, ytM, xbM, ybM))

	porcentagem_manga = round(porcentagem_manga, 2)

	cv2.rectangle(imagem, (int(xtM), int(ytM)), (int(xbM), int(ybM)), (0, 255, 255,), 3)

	#Porcentagem Pedúnculo e Ponto de Corte em imagens 2D
	if(tipo_base == "3D"):

		porcentagem_pedunculo = 0.0

		xtP, ytP, xbP, ybP = coordenadas_conhecidas[1][0][0], coordenadas_conhecidas[1][0][1], coordenadas_conhecidas[1][0][2], coordenadas_conhecidas[1][0][3]
	
		cv2.rectangle(imagem, (int(xtP), int(ytP)), (int(xbP), int(ybP)), (0, 255, 255,), 2)

		xtP_ia, ytP_ia, xbP_ia, ybP_ia = coordenada_pedunculo[0], coordenada_pedunculo[3], coordenada_pedunculo[2], coordenada_pedunculo[1]

		intersecao_pedunculo = calcular_intersecao_porcentagem((xtP_ia, ytP_ia, xbP_ia, ybP_ia), (xtP, ytP, xbP, ybP))

		porcentagem_pedunculo = round(intersecao_pedunculo, 2)

	elif(tipo_base == "2D"):

		porcentagem_ponto = 0

		inclinacao = coordenadas_conhecidas[3]

		#Verifica se o pedúnculo NAO é inclinado
		if(inclinacao == 0):

			xtP, ytP, xbP, ybP = coordenadas_conhecidas[1][0][0], coordenadas_conhecidas[1][0][1], coordenadas_conhecidas[1][0][2], coordenadas_conhecidas[1][0][3]
	
			cv2.rectangle(imagem, (int(xtP - cm_px_2D), int(ytP)), (int(xbP + cm_px_2D), int(ybP)), (0, 255, 255,), 1)

			if((pontoX >= xtP - cm_px_2D) and (pontoX <= xbP + cm_px_2D)):

				porcentagem_ponto = 1

		else:

			encontrou = area_pedunculo(imagem, pontoX, pontoY, coordenadas_conhecidas[1], inclinacao, cm_px_2D)

			if(encontrou):

				porcentagem_ponto = 1

	#Verifica do ponto quando a imagem é 3D
	if(tipo_base == "3D"):

		porcentagem_ponto = 0

		xtPC, ytPC = coordenadas_conhecidas[2][0][0], coordenadas_conhecidas[2][0][1]

		area_px_ponto = encontra_area_px_ponto_corte(fator_ponto_area, fator_cm_px, fator_imgs)

		cv2.circle(imagem, (xtPC, ytPC), 2, (0, 255, 255), -2)
		cv2.circle(imagem, (xtPC, ytPC), area_px_ponto, (0, 255, 255), 3)

		cv2.rectangle(imagem, (int(xtPC - area_px_ponto), int(ytPC - area_px_ponto)), (int(xtPC + area_px_ponto), int(ytPC + area_px_ponto)), (0, 255, 255,), 1)

		if((pontoX >= xtPC - area_px_ponto) and (pontoX <= xtPC + area_px_ponto) and (pontoY >= ytPC - area_px_ponto) and (pontoY <= ytPC + area_px_ponto)):

			porcentagem_ponto = 1

		if(pontoX >= (xtPC - area_px_ponto) and pontoX <= (xtPC + area_px_ponto) and distancia_ponto >= distancia_minima):

			porcentagem_ponto = 1

		distancia_horizontal = round(abs(pontoX - xtPC) * fator_cm_px / fator_imgs, 2)

		distancia_vertical   = round(abs(pontoY - ytPC) * fator_cm_px / fator_imgs, 2)

		distanciaCaixasManga = pc.calcula_distancias_entre_pontos((int(pontoX / fator_imgs), int(pontoY / fator_imgs)), ((int((xtM + xbM) / 2) / fator_imgs), int(ytM / fator_imgs)), "", fator_cm_px)

		cv2.line(imagem, (pontoX, pontoY), (int((xtM + xbM) / 2), int(ytM)), (0, 255, 255), 2)

		distanciaCaixasManga = round(distanciaCaixasManga, 2)

	if(not (caminhoSalva is None)):

		cv2.imwrite(caminhoSalva + str(id_imagem) + "_k_validacao" + ".jpg", imagem)

	return porcentagem_manga, porcentagem_pedunculo, porcentagem_ponto, distancia_horizontal, distancia_vertical, distanciaCaixasManga
