from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('modelo_random_forest_top6.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los valores del formulario y convertir a float
        hiv = float(request.form['HIV/AIDS'])
        income = float(request.form['Income composition of resources'])
        mortality = float(request.form['Adult Mortality'])
        schooling = float(request.form['Schooling'])
        under5 = float(request.form['under-five deaths'])
        bmi = float(request.form['BMI'])

        # Crear DataFrame con los datos de entrada
        input_data = pd.DataFrame([[hiv, income, mortality, schooling, under5, bmi]],
                                  columns=[
                                      'HIV/AIDS',
                                      'Income composition of resources',
                                      'Adult Mortality',
                                      'Schooling',
                                      'under-five deaths',
                                      'BMI'
                                  ])
        
        app.logger.debug(f'DataFrame de entrada: {input_data}')

        # Realizar la predicción
        prediction = model.predict(input_data)
        app.logger.debug(f'Predicción generada: {prediction[0]}')

        # Retornar resultado
        return jsonify({'prediccion': prediction[0]})
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)
