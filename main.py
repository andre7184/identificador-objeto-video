import cv2

import time

import numpy as np

COLORS = [(0,255,255),(255,255,0),(0,255,0),(255,0,0)] #cores para person, yellow,green,blue,red

class_names = [] # nome da lista
with open("coco.names","r") as f: # abre o arquivo coco.names
    class_names = [cname.strip() for cname in f.readlines()] # le o arquivo e armazena em class_names as linhas

cap = cv2.VideoCapture("animal.mp4") # captura o video

net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg") # monta modelo com os pesos e configurações para o yolov4

model = cv2.dnn_DetectionModel(net) # carrega o modelo montado no yolov4

model.setInputParams(size=(416,416),scale=1/255) # define o tamanho da imagem e escala para o yolov4

objetosEncontrados = {} # lista para armazenar os objetos encontrados
tempoEncontrados = {} # lista para armazenar os tempos dos objetos encontrados
while True: # loop infinito
    x,frame = cap.read() # captura frame do video
    start = time.time() # inicia o tempo
    classes, scores, boxes = model.detect(frame,.5,1) # detecta objetos com os pesos e configurações definidas de yolov4 com a imagem capturada 
    end = time.time() # finaliza o tempo

    for (classid, score, box) in zip(classes, scores, boxes): # percorre as classes, scores e boxes detectadas
        color = COLORS[int(classid) % len(COLORS)] # define a cor da caixa de acordo do nome da classe
        cv2.rectangle(frame, box, color, 2) # desenha a caixa
        label = f"{class_names[classid]}:{score}" # nome da classe e score

        cv2.rectangle(frame, box, color, 2) # desenha a caixa
        cv2.putText(frame, label, (box[0],box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # desenha o nome da classe e o score
        # qual time no video o objeto foi detectado
        timeDetection = end - start
        
        if(class_names[classid] not in objetosEncontrados): # se o nome da classe existir na lista objetosEncontrados
            tempoEncontrados[class_names[classid]] = timeDetection
            objetosEncontrados[class_names[classid]] = 1   # adiciona o nome do objeto encontrado ao final da lista
        else:
            tempoEncontrados[class_names[classid]] += timeDetection
            objetosEncontrados[class_names[classid]] += 1 # se ja existe adiciona +1 ao nome do objeto encontrado
    fps_label = f"FPS: {round((1.0/(end-start)),2)}" # calcula o fps
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)  # desenha o fps

    cv2.imshow("detections",frame) # mostra o video
    if cv2.waitKey(1) == 27: # se pressionar a tecla ESC
        break # para o loop

print(objetosEncontrados)
print(tempoEncontrados)

cap.release() # encerra o video
cv2.destroyAllWindows() # fecha todas as janelas




