import src.train as train
import src.dataset as dataset
import src.model as model

if __name__ == "__main__":
    train.train()

    model.model.evaluate(dataset.x_test, dataset.y_test, verbose=1)
