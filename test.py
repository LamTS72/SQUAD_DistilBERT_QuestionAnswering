from models.predictor import Predictor

pred = Predictor()
question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris, known for its art, fashion, and culture."
print(pred.predict(question, context))