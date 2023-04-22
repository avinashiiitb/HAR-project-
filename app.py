from flask import Flask, request,render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.pipeline.predict_pipeline import Predictpipeline
from src.pipeline.predict_pipeline import Preprocessingpipeline
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = Predictpipeline()
path = 'C:\\Users\\avina\\HAR_END_TO_END\\model_128_STRATIFY_ 112_val_accuracy0.93846_val_loss0.31000.h5'
loaded_model = model.give_model(path) 
obj = Preprocessingpipeline()
##route for home page


@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/about")
def about_page():
	return "Human Activity Recognition!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		preds = obj.preprocessing(img_path, loaded_model)

	return render_template("index.html", prediction = preds, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
