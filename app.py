import datetime
from flask import Flask, request
from flask_restx import Resource, Api
from prediction.test_regression import predict
import os
import csv
from preprocessing import crop_image, detect_script
from uuid import uuid4

app = Flask(__name__)
api = Api(app)


@api.route('/image')
class Image(Resource):
    def post(self):
        f = request.files["test"]
        f.save("test.jpg")
        return 0


@api.route('/prediction')
class Prediction(Resource):
    def post(self):
        file = request.files["image"]
        file_name = datetime.datetime.now().isoformat().replace(":", "-") + ".jpg"
        file.save(file_name)
        result = predict(file_name)
        os.remove(file_name)
        return result


@api.route('/detect')
class Detect(Resource):
    def post(self):
        file = request.files["image"]
        name = datetime.datetime.now().isoformat().replace(":", "-").replace(".", ",")
        file_name = name + ".jpg"
        file.save(file_name)
        detect_script.run("preprocessing/best.pt", file_name, save_txt=True, imgsz=[640, 480])
        crop_image.run(file_name, f"results/labels/{name}.txt")
        return 9


@api.route('/predict')
class Predict(Resource):
    def post(self):
        file = request.files["image"]
        name = str(uuid4())
        file_name = "input/" + name + ".jpg"
        file.save(file_name)
        detect_script.run("preprocessing/best.pt", file_name, save_txt=True, imgsz=[640, 512])
        outputs = crop_image.run(file_name, f"results/labels/{name}.txt")
        predictions = [predict(output) for output in outputs]
        result = sum(predictions) / len(predictions)
        with open('results.csv', 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([name, result])
        return result


if __name__ == '__main__':
    app.run()
