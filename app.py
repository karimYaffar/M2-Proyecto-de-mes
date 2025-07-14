from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos del Titanic
model = joblib.load('modelo_knn.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
app.logger.debug('Modelo KNN, scaler y PCA cargados correctamente.')

# Características exactas que usaste en el entrenamiento
features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        sex_female = 1 if request.form['sex'] == 'female' else 0
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        pclass = int(request.form['pclass'])
        cabin = 1 if request.form['cabin'] == 'yes' else 0
        
        # Crear array con los valores en el orden EXACTO del entrenamiento
        input_vals = [sex_female, age, fare, pclass, cabin]
        
        # Crear DataFrame con las características exactas
        X = pd.DataFrame([input_vals], columns=features)
        app.logger.debug(f'DataFrame creado: {X}')
        app.logger.debug(f'Columnas: {list(X.columns)}')
        app.logger.debug(f'Valores: {X.values[0]}')

        # Pipeline exacto: scaler → PCA → modelo
        # 1. Escalar los datos
        X_scaled = scaler.transform(X)
        app.logger.debug(f'Datos escalados - forma: {X_scaled.shape}')
        
        # 2. Aplicar PCA
        X_pca = pca.transform(X_scaled)
        app.logger.debug(f'Datos después de PCA - forma: {X_pca.shape}')

        # 3. Predicción con KNN
        prediction = model.predict(X_pca)
        prediction_proba = model.predict_proba(X_pca)
        
        app.logger.debug(f'Predicción: {prediction[0]}')
        app.logger.debug(f'Probabilidades: {prediction_proba[0]}')

        # Interpretar resultado
        survived = bool(prediction[0])
        probability = float(prediction_proba[0][1])  # Probabilidad de supervivencia
        
        return jsonify({
            'survived': survived,
            'probability': probability,
            'message': 'Sobrevive' if survived else 'No sobrevive'
        })
    
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        import traceback
        app.logger.error(f'Traceback completo: {traceback.format_exc()}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
