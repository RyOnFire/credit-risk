import torch
import torch.nn as nn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from data import load_data, prepare_data

DEFAULT_EPOCHS = 200
DEFAULT_THRESHOLD = 0.75


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_model_with_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_balanced, y_balanced)
    return model


class CreditRiskNet(nn.Module):
    def __init__(self, input_size, hidden_size=16):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


def train_neural_network(X_train, y_train, model_class=CreditRiskNet, epochs=DEFAULT_EPOCHS, **model_kwargs):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    X_train_t = torch.FloatTensor(X_balanced.copy())
    y_train_t = torch.FloatTensor(y_balanced.values.copy()).reshape(-1, 1)

    model = model_class(input_size=X_train.shape[1], **model_kwargs)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model


def predict_with_threshold(model, X, threshold=DEFAULT_THRESHOLD):
    X_t = torch.FloatTensor(X.copy())
    with torch.no_grad():
        probs = model(X_t).numpy().flatten()
    preds = (probs >= threshold).astype(int)
    return preds, probs


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    print("Training Logistic Regression with SMOTE...")
    smote_model = train_model_with_smote(X_train, y_train)
    print("Logistic Regression + SMOTE Results:")
    print(classification_report(y_test, smote_model.predict(X_test)))

    print(f"Training Neural Network ({DEFAULT_EPOCHS} epochs, SMOTE)...")
    nn_model = train_neural_network(X_train, y_train, model_class=CreditRiskNet, hidden_size=16)
    preds, _ = predict_with_threshold(nn_model, X_test)
    print(f"Neural Network Results (threshold={DEFAULT_THRESHOLD}):")
    print(classification_report(y_test, preds))

    joblib.dump(smote_model, 'logistic_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    torch.save(nn_model.state_dict(), 'neural_network_simple.pt')
    print("Models and scaler saved!")
