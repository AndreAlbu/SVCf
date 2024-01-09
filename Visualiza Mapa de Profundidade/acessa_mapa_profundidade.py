import cv2
import numpy as np
import json

#Funcção para organizar o json
def organizaJSON(arquivo):
    
    with open(arquivo, 'r') as file:

        dados = json.load(file)

    lidar_data = dados["LIDARData"]

    return lidar_data

#Função para retornar a distância de um ponto
def measureDistanceOnePoint(ponto1, pontos):
        
    ponto = float(pontos[ponto1[1]][ponto1[0]])
        
    ponto = (ponto * 100)
    
    return ponto

#Função para calcular a distância entre dois pontos
def measureDistance(ponto1, ponto2, pontos):

    xGrau = 81.27321528320013 * ponto1[1] / 256
    yGrau = 57.87424017358846 * ponto1[0] / 192
    ponto1Grau = [yGrau, xGrau]

    xGrau = 81.86114541269245 * ponto2[1] / 256
    yGrau = 58.633679207383835 * ponto2[0] / 192
    ponto2Grau = [yGrau, xGrau]

    angle = np.sqrt((ponto2Grau[0] - ponto1Grau[0]) ** 2 + (ponto2Grau[1] - ponto1Grau[1]) ** 2)
    
    angle = round(angle, 2)

    x = float(pontos[ponto1[1]][ponto1[0]])

    y = float(pontos[ponto2[1]][ponto2[0]])
    
    #Calcula distancia entre os pontos
    size = np.sqrt(x ** 2 + y ** 2 - (2 * x * y * np.cos(np.deg2rad(angle))))
    
    return size

#Função para dá zoom na imagem
def zoom_image(event, x, y, flags, param):
    
    global zoom_factor, zoomed_image, click_x, click_y, click_aux_x, click_aux_y, id_cor
    
    id_cor = 0
    
    click_x, click_y, click_aux_x, click_aux_y = -1, -1, -1, -1

    if event == cv2.EVENT_MOUSEWHEEL:

        delta = flags

        if delta > 0:

            zoom_factor *= 1.1

        else:

            zoom_factor /= 1.1

        zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor)

        cv2.imshow('Imagem', zoomed_image)
        
#Função para obter o fator cm/px da imagem
def obter_cm_px(numero_raios, tamanho, x, y, pontosJson, zoom_factor):
    
    global zoomed_image
    
    cores = [(25,25,112), (30,144,255), (0,250,154), (0,100,0), (127,255,0), (139,69,19), (244,164,96), (216,191,216), (175,238,238)]   
    
    increm = 1
    raioFinal = 0
    
    somaCM = 0
    somaPX = 0
    
    while(increm <= numero_raios):
        
        raioFinal = increm * tamanho
        
        x_1 = x - raioFinal
        x1  = x + raioFinal
        
        distancias = measureDistance((x_1, y), (x1, y), pontosJson)
        
        cv2.circle(zoomed_image, 
                   (int(click_x_r * zoom_factor), 
                    int(click_y_r * zoom_factor)), 
                    int(raioFinal * zoom_factor), 
                    cores[increm])
        
        cv2.circle(zoomed_image, 
                   (int(x_1 * zoom_factor), 
                    int(y * zoom_factor)), 
                    int(2), 
                    (0, 0, 255))
        
        cv2.circle(zoomed_image, 
                   (int(x1 * zoom_factor), 
                    int(y * zoom_factor)), 
                    int(2), 
                    (0, 0, 255))

        somaCM += distancias
        somaPX += (raioFinal * 2)
        
        increm += 1
        
    cm_px = (somaCM / somaPX) * 100
    
    return cm_px

def fator_pixel(event, x, y, flags, param):
    
    global zoomed_image, zoom_factor, click_x_r, click_y_r, raioPixel, pontosJson, cm_px, posicaoTextoY
    
    f_pixel = 0
    
    if event == cv2.EVENT_LBUTTONDOWN:
    
        click_x_r = x
        click_y_r = y

        cv2.circle(zoomed_image, (x, y), raio, (255, 0, 0), -1)

        click_x_r = int(click_x_r / zoom_factor)
        click_y_r = int(click_y_r / zoom_factor)

        cm_px = obter_cm_px(4, 2, click_x_r, click_y_r, pontosJson, zoom_factor)
        
        cv2.putText(img_notas,"cm_px: " + str(round(cm_px, 8)), (10, posicaoTextoY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 1)
    
        posicaoTextoY += 20
    
        cv2.imshow("Imagem", zoomed_image)
        cv2.imshow("Notas", img_notas)
        
        print(f"Fator cm_px: {cm_px}")

        return cm_px

#Função para calcular a distância entre dois pontos com o fator cm/px
def calcula_distancias(ponto1, ponto2, pontosJson):
    
    global cm_px
    
    distancia_p1_p2 = (( (ponto1[0] - ponto2[0]) ** 2) + ((ponto1[1] - ponto2[1]) ** 2)) ** 0.5
    
    #print(distancia_p1_p2)
    
    distancia_final = distancia_p1_p2 * cm_px
    
    return distancia_final

#Função para retornar a distâncias de varias coordenadas
def coordenada_pontos(event, x, y, flags, param):
    
    global click_x, click_y, click_aux_x, click_aux_y, zoom_factor, zoomed_image, id_cor, img_notas, posicaoTextoY, cm_px
    
    cores = [(25,25,112), (30,144,255), (0,250,154), (0,100,0), (127,255,0), (139,69,19), (244,164,96), (216,191,216), (175,238,238)]   
    
    if event == cv2.EVENT_LBUTTONDOWN:

        click_x = x
        click_y = y

        cv2.circle(zoomed_image, (x, y), raio, (255, 0, 0), -1)
        
        click_x = int(click_x / zoom_factor)
        click_y = int(click_y / zoom_factor)
        
        ponto1 = measureDistanceOnePoint((click_x, click_y), pontosJson)
        
        cv2.putText(img_notas,"Ponto 1: " + str(ponto1) + " cm", (10, posicaoTextoY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, 1)
        
        print(f"Distancia P1: ({click_x}, {click_y}) -- {ponto1}")
        
        posicaoTextoY += 20

    if event == cv2.EVENT_RBUTTONDOWN:
        
        if(id_cor == len(cores)):
            
            id_cor = 0
    
        click_aux_x = x
        click_aux_y = y

        cv2.circle(zoomed_image, (click_aux_x, click_aux_y), raio, (0,0,255), -1)
        
        click_aux_x = int(click_aux_x / zoom_factor)
        click_aux_y = int(click_aux_y / zoom_factor)
        
        if(click_x > 0 and click_y > 0):
            
            #posicaoTextoY += 20
            
            if(cm_px >= 0):
            
                ponto2 = measureDistanceOnePoint((click_aux_x, click_aux_y), pontosJson)

                distancia = calcula_distancias((click_x, click_y), (click_aux_x, click_aux_y), pontosJson)

                print(f"Distancia P2: ({click_aux_x}, {click_aux_y}) -- {round(ponto2, 2)}")
                print(f"Distancia P1 <---> P2: {distancia}")

                cv2.putText(img_notas,"Ponto 2: " + str(round(ponto2, 2)) + " cm", (10, posicaoTextoY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 1)

                posicaoTextoY += 20

                cv2.putText(img_notas,"Distancia: " + str(round(distancia, 2)) + " cm", (10, posicaoTextoY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 1)

                posicaoTextoY += 10

                cv2.putText(img_notas,"---", (10, posicaoTextoY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, 1)

                posicaoTextoY += 20

                cv2.line(zoomed_image, (int(click_x * zoom_factor), int(click_y * zoom_factor)), (int(click_aux_x * zoom_factor), int(click_aux_y * zoom_factor)), cores[id_cor], line)

                id_cor += 1
                
            else:
                
                print("cm_px não foi definido!")
            
    cv2.imshow('Imagem', zoomed_image)
        
    cv2.imshow("Notas", img_notas)
    
def marca_ponto_conhecido(coordenadas):
    
    global zoomed_image, zoom_factor, pontosJson
        
    for coord in coordenadas:
        
        cv2.circle(zoomed_image, (int(coord[0] * zoom_factor), int(coord[1] * zoom_factor)), 1, (0, 0, 0), -1)
        
        ditancia = measureDistanceOnePoint((coord[0], coord[1]), pontosJson)
        
        print(f"Distancia P: {(coord[0], coord[1])}: {ditancia}")
        
    cv2.imshow('Imagem', zoomed_image)
    
def visualiza_distancias_proximas(limite_inferior, limite_superior, pontosJson, imagem_original):
    
    altura, largura = imagem_original.shape[:2]
    
    distanciasCorretas = []
    pontosCorretos     = []
    
    imagem_vizualizacao = np.copy(imagem_original)

    for y in range(altura):
        
        for x in range(largura):
            
            distanciaPonto = measureDistanceOnePoint((x, y), pontosJson)
            
            if(distanciaPonto > limite_inferior and distanciaPonto < limite_superior):
                
                cv2.circle(imagem_vizualizacao, (x,y), 1, (255,0,0))
                
                distanciasCorretas.append(distanciaPonto)
                
    imagem_vizualizacao = cv2.resize(imagem_vizualizacao, (384, 512))
                
    cv2.imshow("Vizualiza", imagem_vizualizacao)
    
global click_x, click_y, pontosJson, image, zoom_factor, zoomed_image, id_cor, posicaoTextoY, valorPixel, raioPixel, f_pixel, cm_px

zoom_factor = 1.1
id_cor = 0
raio = 1
line = 1
raioPixel = 4
posicaoTextoY = 20

#CONFIGURAÇÕES
QUANTIDADE_IMAGEM = 5
PROFUNDIDADE = False
PASTA = "base_26_11_2023/"
SUBPASTA = "(2)/"

LIMITE_INFERIOR = 15
LIMITE_SUPERIOR = 80
    
MARCAR_PONTOS_AUXILIARES = False
COORDENADAS = [(94,132), (89,132)]
        
def main():
    
    global inicio, image, pontosJson, zoomed_image, img_notas, posicaoTextoY, cm_px
    
    altura = 192
    largura = 256
    
    idx = 1
    
    while(idx <= QUANTIDADE_IMAGEM):
        
        print(f"Imagem: {idx}")
        
        if(PROFUNDIDADE):
            
            IMAGEM = str(idx) + "_ImageDepth.jpg"
            
        else:
            
            IMAGEM = str(idx) + ".jpg"
    
        caminho = PASTA + SUBPASTA + IMAGEM
        
        print(f"Caminho imagem -> {caminho}")
        print(f"Caminho json ->   {PASTA + SUBPASTA + str(idx)}.json")

        image = cv2.imread(PASTA + SUBPASTA + IMAGEM)
        image = cv2.resize(image, (altura, largura))

        pontosJson = organizaJSON(PASTA + SUBPASTA + str(idx) + ".json")
        
        img_notas = np.zeros((900, largura, 3), dtype=np.uint8)

        cv2.imshow("Notas", img_notas)

        cv2.namedWindow('Imagem')
        cv2.imshow('Imagem', image)

        zoomed_image = np.copy(image)

        cv2.imshow('Imagem', zoomed_image)

        inicio = zoom_image

        cv2.setMouseCallback('Imagem', inicio)

        while True:

            key = cv2.waitKey(0)

            if key == 27:
                
                idx = QUANTIDADE_IMAGEM + 1

                break

            elif key == ord('z'):

                cv2.setMouseCallback('Imagem', zoom_image)

                if(MARCAR_PONTOS_AUXILIARES):

                    marca_ponto_conhecido(COORDENADAS)

            elif key == ord('f'):

                cv2.setMouseCallback('Imagem', fator_pixel)

            elif key == ord('d'):

                cv2.setMouseCallback('Imagem', coordenada_pontos)
                
            elif key == ord('p'):
                
                idx += 1
                cv2.destroyAllWindows()
                click_x = 0
                click_y = 0 
                pontosJson = 0 
                image = 0 
                zoom_factor = 1.0 
                zoomed_image = 0 
                id_cor = 0 
                posicaoTextoY = 20 
                valorPixel = 0 
                raioPixel = 4 
                f_pixel = 0 
                cm_px = -1
                print("-----------------------------------------")
                
                break
                
            elif key == ord('v'):
                
                visualiza_distancias_proximas(LIMITE_INFERIOR, LIMITE_SUPERIOR, pontosJson, image)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()