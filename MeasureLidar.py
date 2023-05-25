import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def resizeImage(image, divider):
    dim = (int(image.shape[1] / divider), int(image.shape[0] / divider))
    image_resized = cv.resize(image, dim, interpolation=cv.INTER_LINEAR)
    return image_resized


def mostra(legenda, imagem):
    cv.imshow(legenda, imagem)
    cv.waitKey(0)
    return imageResized


def showTwoImages(image1, image2):
    fig = plt.figure(figsize=(10, 7))

    rows = 1
    columns = 2

    fig.add_subplot(rows, columns, 1)

    plt.imshow(image1)
    # plt.axis('off')
    plt.title("Original reduzida")

    fig.add_subplot(rows, columns, 2)

    plt.imshow(image2)
    # plt.axis('off')
    plt.title("Representação Lidar")
    plt.show()
    cv.waitKey(0)


def organizaTXT(arquivo):
    arq = open(arquivo)
    lines = arq.readlines()

    pontos = []
    print("Number of lines = " + str(len(lines)))
    for i in range(len(lines)):
        values = lines[i].split(",")
        print("Line " + str(i) + " there are = " + str(len(values)))
        for j in range(len(values)):
            print(str(j) + " : " + values[j])
        pontos.append(values)
    # print(pontos)
    return pontos


def calibrationLIDAR(pontos, imageOriginal, imageLidar):
    ## Eixo vertical
    ponto1 = [0, 127]
    ponto2 = [191, 127]

    ## Eixo horizontal
    # ponto1 = [95, 0]
    # ponto2 = [95, 255]

    sizeHeight = 0.168
    sizeWidth = 0.248

    x = float(pontos[ponto1[0]][ponto1[1]])
    print(str(x) + " m")
    y = float(pontos[ponto2[0]][ponto2[1]])
    print(str(y) + " m")

    angle = np.arccos((x**2+y**2-sizeHeight**2)/(2*x*y))
    print(angle)
    print(np.rad2deg(angle))
    ## valor achado vertical 57.87424017358846
    ## valor achado horizontal 81.27321528320013

    # a = 58.24929668917284
    # b = 59.01806172559483
    # media = (a + b) / 2


    # angle = np.arccos((x ** 2 + y ** 2 - sizeWidth ** 2) / (2 * x * y))
    # print(angle)
    # print(np.rad2deg(angle))
    # a = 81.40737774694823
    # b = 82.31491307843667
    # media = (a + b) / 2
    # print(media)

    cv.circle(imageOriginal, (ponto1[1], ponto1[0]), 0, (0, 0, 255), -1)
    cv.circle(imageOriginal, (ponto2[1], ponto2[0]), 0, (255, 0, 0), -1)
    cv.circle(imageLidar, (ponto1[1], ponto1[0]), 0, (0, 0, 255), -1)
    cv.circle(imageLidar, (ponto2[1], ponto2[0]), 0, (255, 0, 0), -1)
    showTwoImages(imageOriginal, imageLidar)


def measureDistance(ponto1, ponto2, pontos, image, imageLidar):

    xGrau = 81.27321528320013 * ponto1[1] / 256
    yGrau = 57.87424017358846 * ponto1[0] / 192
    ponto1Grau = [yGrau, xGrau]

    xGrau = 81.86114541269245 * ponto2[1] / 256
    yGrau = 58.633679207383835 * ponto2[0] / 192
    ponto2Grau = [yGrau, xGrau]

    angle = np.sqrt((ponto2Grau[0] - ponto1Grau[0]) ** 2 + (ponto2Grau[1] - ponto1Grau[1]) ** 2)
    print("O angulo encontrado é: " + str(angle) + " graus")

    x = float(pontos[ponto1[0]][ponto1[1]])
    print(str(x) + " m")
    y = float(pontos[ponto2[0]][ponto2[1]])
    print(str(y) + " m")

    size = np.sqrt(x ** 2 + y ** 2 - (2 * x * y * np.cos(np.deg2rad(angle))))
    print("O valor calculado é: " + str(size * 100) + " cm")

    cv.circle(image, (ponto1[1], ponto1[0]), 0, (0, 0, 255), -1)
    cv.circle(image, (ponto2[1], ponto2[0]), 0, (255, 0, 0), -1)
    cv.circle(imageLidar, (ponto1[1], ponto1[0]), 0, (0, 0, 255), -1)
    cv.circle(imageLidar, (ponto2[1], ponto2[0]), 0, (255, 0, 0), -1)
    showTwoImages(image, imageLidar)



def main():
    import matplotlib
    matplotlib.use('MacOSX')

    baseURL = "/Users/alanpaulino/projects/NewCapture/Demo/"
    numberImage = "3"

    pontos = organizaTXT(baseURL + numberImage + "_depth.txt")

    imageOriginal = cv.imread(baseURL + numberImage + ".jpg")
    image = resizeImage(imageOriginal, 7.5)

    imageLidar = cv.imread(baseURL + numberImage + "_imageDepth.jpg")

    ponto1 = [102, 115]
    ponto2 = [96, 123]

    # ponto1 = [75, 129]
    # ponto2 = [82, 129]

    # calibrationLIDAR(pontos, image, imageLidar)
    measureDistance(ponto1, ponto2, pontos, image, imageLidar)



if __name__ == "__main__":
    main()
