import pickle
import warnings

warnings.filterwarnings('ignore')
def load_model(filename):
    """ Load the trained model from a pickle file. """
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_with_model(model, user_input):
    """ Make a prediction using the model and user input. """
    prediction = model.predict([user_input])
    return prediction[0]

def main():
    model_filename = 'xgb_model.pkl'
    model = load_model(model_filename)

    user_input = [5.1, 3.5, 1.4, 0.2]
    prediction = predict_with_model(model, user_input)
    print(f"The predicted output is: {prediction}")

if __name__ == "__main__":
    main()