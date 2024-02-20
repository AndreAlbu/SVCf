import json

import Lidar as lidar
import PC_SVCF as pc

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR ÁREA DA MANGA    	     					#
#																									#
#####################################################################################################

def ler_coordenadas_conhecidas(id_imagem, ARQUIVO_JSON, TIPO_BASE):

    with open(ARQUIVO_JSON) as file:

        coordenadas = json.load(file)

    informacao_imagem = coordenadas.get('InformationImage')

    if informacao_imagem is None:

        print(f"{id_imagem}: não existe informações de posição das caixas")

        print("<-------------------------------------->")

        return

    REDUCAO_IMAGEM   = coordenadas['InformationImage']['reduction_value']
    QUANTIDADE_MANGA = coordenadas['InformationImage']['mango_quantity']
    
    REDUCAO_IMAGEM = 1
    
    coordenadas_manga     = [ ]
    coordenadas_pedunculo = [ ]
    coordenadas_ponto     = [ ]

    for idx_manga in range(QUANTIDADE_MANGA):

        #Coordenadas da manga
        xt, yt = coordenadas['MangoCoordinates_' + str(idx_manga)]['xt'], coordenadas['MangoCoordinates_' + str(idx_manga)]['yt']
        xb, yb = coordenadas['MangoCoordinates_' + str(idx_manga)]['xb'], coordenadas['MangoCoordinates_' + str(idx_manga)]['yb']
        coordenadas_manga.append((xt, yt, xb, yb))
        
        #Coordenadas do pedúnculo
        xt, yt = coordenadas['PeduncleCoordinates_' + str(idx_manga)]['xt'], coordenadas['PeduncleCoordinates_' + str(idx_manga)]['yt']
        xb, yb = coordenadas['PeduncleCoordinates_' + str(idx_manga)]['xb'], coordenadas['PeduncleCoordinates_' + str(idx_manga)]['yb']
        coordenadas_pedunculo.append((xt, yt, xb, yb))
        
        #Coordenadas do ponto
        if(TIPO_BASE == "3D"):

            xt, yt = coordenadas['PointCoordinates_' + str(idx_manga)]['xt'], coordenadas['PointCoordinates_' + str(idx_manga)]['yt']
            coordenadas_ponto.append((xt, yt))
    
    return coordenadas_manga, coordenadas_pedunculo, coordenadas_ponto

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

#####################################################################################################
#																									#
#							      FUNÇÕES PARA VALIDAR PONTO DE CORTE    	     					#
#																									#
#####################################################################################################

def encontra_area_px_ponto_corte(coord, cm_px, fator_imgs):
        
    raio = math.sqrt(coord / math.pi)
    
    area_px = raio / cm_px * fator_imgs

    return round(area_px)

#####################################################################################################
#																									#
#							                    FUNCAO PRINCIPAL    								#
#																									#
#####################################################################################################

def valida_svcf(id_imagem, tipo_base, coordenada_manga_ia, coordenada_pedunculo, arquivo_json, fator_cm_px):

	coordenadas_conhecidas = ler_coordenadas_conhecidas(id_imagem, arquivo_json, tipo_base)

	cm_px_2D = 37.795

	xtM, ytM, xbM, ybM = coordenadas_conhecidas[0][0][0], coordenadas_conhecidas[0][0][1], coordenadas_conhecidas[0][0][2], coordenadas_conhecidas[0][0][3]

	xtP, ytP, xbP, ybP = coordenadas_conhecidas[1][0][0], coordenadas_conhecidas[1][0][1], coordenadas_conhecidas[1][0][2], coordenadas_conhecidas[1][0][3]
	
	if(tipo_base == "3D"):

		xtPC, ytPC, xbPC, ybPC = coordenadas_conhecidas[2][0][0], coordenadas_conhecidas[1][0][1]

	porcentagem_manga = porcentagem_intersecao_area_manga(coordenada_manga_ia, (xtM, ytM, xbM, ybM))

	porcentagem_manga = round(porcentagem_manga, 2)

	porcentagem_pedunculo = 0.0

	if(xtP >= coordenada_pedunculo[0] and xbP <= coordenada_pedunculo[2]):

		porcentagem_pedunculo = 100

	if(tipo_base == "3D"):

		print("Base 3D")

	elif(tipo_base == "2D"):



	print(porcentagem_manga)
	print(porcentagem_pedunculo)

