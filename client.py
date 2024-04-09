import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from data_prepare import prepare_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        total_loss = 0.0
        total_count = 0
        correct_count = 0
        for x, y in trainloader:
            x, y = x.to(torch.float32).to(DEVICE), y.to(torch.long).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_count += y.size(0)
            correct_count += (predicted == y).sum().item()
        average_loss = total_loss / total_count
        accuracy = correct_count / total_count
        print(f">>> Local training - Epoch {epoch+1}/{epochs} - Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return average_loss, accuracy


def test(model, testloader):
    """Test the model on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0
    correct_count = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(torch.float32).to(DEVICE), y.to(torch.long).to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item() * y.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_count += y.size(0)
            correct_count += (predicted == y).sum().item()
    average_loss = total_loss / total_count
    accuracy = correct_count / total_count
    print(f">>> Local testing - Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy


class MyModel(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


args = argparse.ArgumentParser()

# Add the arguments to the parser
args.add_argument("--n", type=int, default=1000)
args.add_argument("--simulate_seed", type=int, default=42)
args.add_argument("--split_seed", type=int, default=42)
args.add_argument("--batch_size", type=int, default=16)
args = args.parse_args()


# Initialize the model and data
num_features = 10
num_hidden = 32
net = MyModel(num_features, num_hidden).to(DEVICE)
trainloader, testloader, num_examples = prepare_data(
    n=args.n,
    simulate_seed=args.simulate_seed,
    split_seed=args.split_seed,
    batch_size=args.batch_size
    )


class MyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        # print("get_parameters called")
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        # print("set_parameters called")
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # print("fit called")
        self.set_parameters(parameters)
        average_loss, accuracy = train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {"accuracy": float(accuracy), "loss": float(average_loss)}

    def evaluate(self, parameters, config):
        # print("evaluate called")
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}
    
    
fl.client.start_client(server_address="127.0.0.1:8080", client=MyClient().to_client())