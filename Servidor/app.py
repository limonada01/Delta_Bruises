from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
import os
import tensorflow as tf
from flask_cors import CORS
from PIL import Image
import json


app = Flask(__name__)
CORS(app)


def preprocesar_img(img):
    img_size=150
    img = img.resize((img_size,img_size))#Hace el reescalado de la imagen a 250x250
    img = img.convert('L')#Convierte la imagen a escala de grises
    img = np.asarray(img)#transforma la img a matriz numpy
    #NORMALIZAR IMAGEN
    img = normalizar(img)
    img = np.asarray([img])#agrega una dimension tamaño del batch, el cual es 1 ya que solo es una imagen
    return img

#FUNCION NORMALIZAR
def normalizar(imagen):
  imagen = tf.cast(imagen, tf.float32)#casteo los valores a float
  imagen /= 255 #Aqui se pasa de 0-255 a 0-1
  return imagen


# Carga tu modelo entrenado y sus pesos
modelo = keras.models.load_model('Resources/ModeloCNN.h5')

@app.route('/predict', methods=['POST'])
def predecir():
    try:
        # Verifica si la solicitud tiene un archivo adjunto
        if 'imagen' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'})

        archivo_imagen = request.files['imagen']

        # Verifica si el archivo tiene un nombre
        if archivo_imagen.filename == '':
            return jsonify({'error': 'Nombre de archivo no válido'})
        # Verifica la extensión del archivo
        nombre, extension = os.path.splitext(archivo_imagen.filename) #divide el txt en el ultimo punto, es decir el nombre de la extension
        if extension.lower() not in ['.jpg', '.jpeg']:
            return jsonify({'error': 'La imagen debe tener la extensión .jpg o .jpeg'})

        # Guarda la imagen en el servidor o realiza el procesamiento que desees
        #ruta_guardado = '/Resources/Img/' +archivo_imagen.filename
        #archivo_imagen.save(ruta_guardado)
        #print("FLAAAAGsss")
        # Aquí puedes realizar cualquier procesamiento adicional con la imagen si es necesario
        img = Image.open(archivo_imagen)
        img = preprocesar_img(img)
        resultado = modelo.predict(img)
        #print(archivo_imagen)
        #eliminar imagen recibida
        #os.remove(ruta_guardado)
        
        print("RESULTADO: ",float(resultado[0][0]))
        
       
        # Serializar la lista a JSON
        resultado_json = json.dumps({'mensaje':float(resultado[0][0])})
        print(resultado_json)
        return jsonify(resultado_json)
    except Exception as e:
        print('error: ',str(e))
        return jsonify({'Exception error': str(e)})
    
    

@app.route('/')
def home():
    return "hola mundo!"

if __name__ == '__main__':
    app.run(debug=True)






