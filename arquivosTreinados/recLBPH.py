#coding: utf-8
import cv2

camera = cv2.VideoCapture(0)
detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecimentoFace = cv2.face.LBPHFaceRecognizer_create()
reconhecimentoFace.read("classificadorLBPH.yml")
largura, altura = 220, 220
tagFont = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:

	conectado, imagem = camera.read()
	
	imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
	facesDetectadas = detectorFace.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(30, 30))
	
	for (x,y,l,a) in facesDetectadas:
		imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
		cv2.rectangle(imagem, (x, y),(x + l, y + a),(0,0,255), 2)

		#reconhecimento
		ids, autenticador = reconhecimentoFace.predict(imagemFace)
		if ids == 1:
			nome = 'Erlon'
		else:
			nome = 'Nao encontrado'
		cv2.putText(imagem, nome, (x, y+(a+30)), tagFont, 2, (0,0,255))
		#cv2.putText(imagem, str(autenticador), (x,y + (a+50)), tagFont, 1,(0,255,0))
		
	
	cv2.imshow("Face", imagem)
	
	if cv2.waitKey(1) == ord('q'):
		break
	
camera.release()
cv2.destroyAllWindows()
