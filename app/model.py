import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

class DenseNetwork(nn.Module):
    def __init__(self):
        super(DenseNetwork, self).__init__()
        self.fc1 = nn.Linear(3000, 1000)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(1000, 500)
        self.drop2 = nn.Dropout(0.4)
        self.prediction = nn.Linear(500, 25)

    def forward(self, x):
        x = F.relu(self.fc1(x.to(torch.float)))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = F.log_softmax(self.prediction(x), dim=1)
        return x


label_y = {
    0: 'Data Science',
    1: 'HR',
    2: 'Advocate',
    3: 'Arts',
    4: 'Web Designing',
    5: 'Mechanical Engineer',
    6: 'Sales',
    7: 'Health and fitness',
    8: 'Civil Engineer',
    9: 'Java Developer',
    10: 'Business Analyst',
    11: 'SAP Developer',
    12: 'Automation Testing',
    13: 'Electrical Engineering',
    14: 'Operations Manager',
    15: 'Python Developer',
    16: 'DevOps Engineer',
    17: 'Network Security Engineer',
    18: 'PMO',
    19: 'Database',
    20: 'Hadoop',
    21: 'ETL Developer',
    22: 'DotNet Developer',
    23: 'Blockchain',
    24: 'Testing'
}



def load_model_and_vectorizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "model", "classifier.pth")
    vectorizer_path = os.path.join(base_dir, "model", "vectorizer.pkl")

    model = DenseNetwork()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer, label_y, device