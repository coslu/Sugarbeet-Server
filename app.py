import math
from flask import Flask, request
from flask_restx import Resource, Api
from prediction.test_regression import predict
import os
from preprocessing import crop_image, detect_script
from google.cloud import storage

app = Flask(__name__)
api = Api(app)

# Change model path here
prediction_model = "prediction/model_300epochs.pth"
preprocessing_model = "preprocessing/best.pt"


@api.route('/predict')
class Predict(Resource):
    def post(self):
        file = request.files["image"]
        name = file.filename.removesuffix('.jpg')
        file_name = "input/" + name + ".jpg"
        file.save(file_name)

        # connect to cloud storage
        storage_client = storage.Client()
        bucket = storage_client.bucket("sugarbeetmonitoring_data")

        # save input image
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)

        result = predict(file_name, prediction_model)

        # save result
        blob = bucket.blob("output/" + name + ".txt")
        blob.upload_from_string(str(result))

        return result


@api.route('/process')
class Process(Resource):
    def post(self):
        file = request.files["image"]
        name = file.filename.removesuffix('.jpg')
        file_name = "input/" + name + ".jpg"
        file.save(file_name)

        # connect to cloud storage
        storage_client = storage.Client()
        bucket = storage_client.bucket("sugarbeetmonitoring_data")

        # save input image
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)

        detect_script.run(preprocessing_model, file_name, save_txt=True, imgsz=[640, 640])
        try:
            # get the box that is closest to the center
            lines = open(f"results/labels/{name}.txt").readlines()
            min_dist = math.inf
            min_dist_index = 0
            for i, line in enumerate(lines):
                numbers = line.split(" ")
                dist = distance(float(numbers[1]), float(numbers[2]))
                if dist < min_dist:
                    min_dist = dist
                    min_dist_index = i

            outputs = crop_image.run(file_name, f"results/labels/{name}.txt")
            result = predict(outputs[min_dist_index], prediction_model)
        except Exception:
            # e.g. when there is no box
            result = predict(file_name, prediction_model)

        # save result
        blob = bucket.blob("output/" + name + ".txt")
        blob.upload_from_string(str(result))

        return result


def distance(x, y):
    return (0.5 - x) ** 2 + (0.5 - y) ** 2


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
