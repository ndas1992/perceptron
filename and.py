from utils.model import perceptron


AND = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [0,0,0,1]}

df = pd.DataFrame(AND)


X, y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model_and = Perceptron(eta=ETA, epochs=EPOCHS)
model_and.fit(X, y)

_ = model_and.total_loss()
