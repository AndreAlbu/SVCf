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


def main():
    import matplotlib
    matplotlib.use('MacOSX')

    baseURL = "/Users/alanpaulino/projects/Pack estudo lidar/"
    numberImage = "5"

    pontos = organizaTXT(baseURL + numberImage + "_depth.txt")

    ponto1 = [60, 131]
    ponto2 = [75, 131]

    # sizeHeight = 0.26
    # sizeHeight = 0.465
    # sizeWidth = 0.35
    # sizeWidth = 0.635

    # x = float(pontos[ponto1[0]][ponto1[1]])
    # print(x)
    # y = float(pontos[ponto2[0]][ponto2[1]])
    # print(y)
    # angle = np.arccos((x**2+y**2-sizeHeight**2)/(2*x*y))
    # print(angle)
    # print(np.rad2deg(angle))
    # a = 58.24929668917284
    # b = 59.01806172559483
    # media = (a + b) / 2

    x = float(pontos[ponto1[0]][ponto1[1]])
    print(x)
    y = float(pontos[ponto2[0]][ponto2[1]])
    print(y)
    # angle = np.arccos((x ** 2 + y ** 2 - sizeWidth ** 2) / (2 * x * y))
    # print(angle)
    # print(np.rad2deg(angle))
    # a = 81.40737774694823
    # b = 82.31491307843667
    # media = (a + b) / 2
    # print(media)

    xGrau = 81.86114541269245 * ponto1[1] / 256
    yGrau = 58.633679207383835 * ponto1[0] / 192
    ponto1Grau = [yGrau, xGrau]

    xGrau = 81.86114541269245 * ponto2[1] / 256
    yGrau = 58.633679207383835 * ponto2[0] / 192
    ponto2Grau = [yGrau, xGrau]

    angle = np.sqrt((ponto2Grau[0] - ponto1Grau[0]) ** 2 + (ponto2Grau[1] - ponto1Grau[1]) ** 2)
    print("O angulo encontrado é: " + str(angle))

    size = np.sqrt(x ** 2 + y ** 2 - (2 * x * y * np.cos(np.deg2rad(angle))))
    print("O valor calculado é: " + str(size * 100) + " cm")

    imageOriginal = cv.imread(baseURL + numberImage + ".jpg")
    image = resizeImage(imageOriginal, 7.5)

    imageLidar = cv.imread(baseURL + numberImage + "_imageDepth.jpg")

    cv.circle(image, (ponto1[1], ponto1[0]), 1, (0, 0, 255), -1)
    cv.circle(image, (ponto2[1], ponto2[0]), 1, (255, 0, 0), -1)
    cv.circle(imageLidar, (ponto1[1], ponto1[0]), 0, (0, 0, 255), -1)
    cv.circle(imageLidar, (ponto2[1], ponto2[0]), 0, (255, 0, 0), -1)
    showTwoImages(image, imageLidar)

if __name__ == "__main__":
    main()