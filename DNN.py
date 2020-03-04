import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

EPOCHS = 100
LOSSES = []


class Model(nn.Module):
    def __init__(self, in_features=4, h1=10, h2=12, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)

        self.cat1_l1 = nn.Linear(h2, out_features)
        self.cat2_l1 = nn.Linear(h2, out_features)
        self.cat3_l1 = nn.Linear(h2, out_features)

        self.cat1_l2 = nn.Linear(out_features, out_features)
        self.cat2_l2 = nn.Linear(out_features, out_features)
        self.cat3_l2 = nn.Linear(out_features, out_features)

        self.out1 = nn.Linear(out_features, 1)
        self.out2 = nn.Linear(out_features, 1)
        self.out3 = nn.Linear(out_features, 1)

        self.out = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x1 = F.relu(self.cat1_l1(x))
        x2 = F.relu(self.cat2_l1(x))
        x3 = F.relu(self.cat3_l1(x))

        x1 = F.relu(self.cat1_l2(x1))
        x2 = F.relu(self.cat2_l2(x2))
        x3 = F.relu(self.cat3_l2(x3))

        x1 = self.out1(x1)
        x2 = self.out2(x2)
        x3 = self.out3(x3)

        out = self.out(torch.cat((x1, x2, x3), dim=1))

        return out


if __name__ == "__main__":
    model = Model()
    data = load_iris()
    X = np.array(data["data"])
    y = np.array(data["target"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(EPOCHS):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        LOSSES.append(loss)
        print(f"Epoch {i}, loss = {loss}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
        print(f"testing loss: {loss}")

    torch.save(model.state_dict(), "model.pt")
