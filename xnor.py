from utils.model import Perceptron
from utils.all_utils import prepare_data, save_plot, save_model
import pandas as pd
import numpy as np


def main(data, eta, epochs, model_fname, plot_fname):

    df = pd.DataFrame(data)

    print(df)

    X, y = prepare_data(df)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename=model_fname)
    save_plot(df, file_name=plot_fname, model=model)

if __name__ == "__main__": #entry point

    XNOR = {"x1": [0,0,1,1], "x2": [0,1,0,1], "y": [1,0,0,1]}

    ETA = 0.3 # 0 and 1
    EPOCHS = 10
    
    main(data=XNOR, eta=ETA, epochs=EPOCHS, model_fname="xnor.model", plot_fname="xnor.png")