"""
    Nome do Módulo: Lidar

    Descrição:
      Este módulo contém funções para lidar com dados do sensor Lidar no contexto do Projeto SVCF.

    Autor:
      Hairon Gonçalves
      André Albuquerque
      Alan Paulino

    Data de Criação:
    Criação: 05/01/2024
    Última Modificação: 30/01/2024

    Requisitos:
        Bibliotecas:
           - Opencv
           - Numpy
           - Json
            
    Licença:
      Licença MIT

    Notas:
      Essa biblioteca faz parte do Projeto SVCF

"""

import cv2
import numpy as np
import json


def organizaJSON(arquivo):
    
    """
    Função: Organiza o arquivo Json em linhas e colunas
    Parâmetro: 
        @arquivo: Arquivo no formato Json com as coordenadas
    Retorno:
        @lidar_data: Matriz com as distâncias organizadas em linhas e colunas
    """
    with open(arquivo, 'r') as file:

        dados = json.load(file)

    lidar_data = dados["LIDARData"]

    return lidar_data

def measureDistanceOnePoint(ponto1, pontos):
    
    """
    Função: Realiza medida do objeto para o sensor Lidar
    Parâmetros:
        @ponto1: coordenadas x e y do ponto desejado
        @pontos: matriz de distâncias
    Retorno:
        @distancia: distância em cm do objeto para o sensor Lidar
    """
        
    distancia = float(pontos[ponto1[1]][ponto1[0]])
        
    distancia = (distancia * 100)
    
    return distancia

def measureDistance(ponto1, ponto2, pontos):
    
    """
    Função: Realiza medida entre dois pontos
    Parâmetros:
        @ponto1: coordenadas x e y do ponto 1 desejado
        @ponto2: coordenadas x e y do ponto 2 desejado
        @pontos: matriz de distâncias
    Retorno:
        @distancia: distância em cm de dois pontos
    """

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
    distancia = np.sqrt(x ** 2 + y ** 2 - (2 * x * y * np.cos(np.deg2rad(angle))))
    
    return distancia
