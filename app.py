from flask import Flask, request
from flask_restx import Resource, Api
from prediction.test_regression import predict
import os
from preprocessing import crop_image, detect_script
from uuid import uuid4
from google.cloud import storage

app = Flask(__name__)
api = Api(app)


@api.route('/predict')
class Predict(Resource):
    def post(self):
        file = request.files["image"]
        name = str(uuid4())
        file_name = "input/" + name + ".jpg"
        file.save(file_name)

        storage_client = storage.Client()
        bucket = storage_client.bucket("sugarbeetmonitoring_data")
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)

        detect_script.run("preprocessing/best.pt", file_name, save_txt=True, imgsz=[640, 512])
        outputs = crop_image.run(file_name, f"results/labels/{name}.txt")
        predictions = [predict(output) for output in outputs]
        result = sum(predictions) / len(predictions)
        return result


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
