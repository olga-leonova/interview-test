import os
from flask import Flask
from flask_restful import reqparse, Api, Resource
from interview_test import modeling as ml

app = Flask(__name__)
api = Api(app)

MODEL_PATH = f'{os.getcwd()}/model'

parser = reqparse.RequestParser()
parser.add_argument('text_article')


class ProductIDPredictor(Resource):
    def get(self):
        args = parser.parse_args()
        text_dict = {'text_article': args['text_article']}
        predicted_product_id = ml.generate_single_predictions(text_dict['text_article'], MODEL_PATH)
        return predicted_product_id, 201


api.add_resource(ProductIDPredictor, '/productid')


if __name__ == '__main__':
    app.run()
