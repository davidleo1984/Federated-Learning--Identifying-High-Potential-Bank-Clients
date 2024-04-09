# Federated Learning System: Identifying High-Potential Bank Customers

## Project Overview

This project aims to implement a federated learning system to identify high-potential customers from dummy bank data distributed across multiple simulated branches. It demonstrates the application of federated learning, data privacy, and machine learning algorithms in the finance domain.

## Methodology

### 1. Scenario and Data Preparation

#### Scenario Description
Assume that a bank has multiple branches, each with its own dataset of customer transactions and profiles. Due to privacy regulations, these datasets cannot be centralized.

#### Data Simulation
The `data_prepare.py` script allow user to simulates customer data across at multiple branches (in this project, we simulate 3 braches). Each branch's dataset includes the following features:

- Account balance
- Transaction history
- Loan history
- Demographic information (age, gender, occupation)

To ensure data diversity and realism, the distribution ranges for account balances, transaction volumes, and loan amounts are adjusted based on different occupations. One-hot encoding is applied to categorical features.

Finally, a "High Potential" label is assigned to each customer based on the following criteria:
- Account balance higher than the average
- Transaction amount higher than the average
- Loan repayment amount lower than the average

A customer is marked as "High Potential" if any two of the above criteria are met.

### 2. Federated Learning Setup

The federated learning model is implemented using the Flower framework. It allows training the model on local datasets at each branch without exchanging raw data. Only model updates (e.g., weights, gradients) are shared with a central server for aggregation.

### 3. Model Selection and Training

#### Model Selection
A simple feed-forward neural network with a single hidden layer is chosen as the base model. This model is suitable for handling structured data and binary classification tasks.

```python
class MyModel(nn.Module):
    def __init__(self, num_features, num_hidden):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### Model Training
The `client.py` script defines the following functions:

- `train` function: Trains the model using cross-entropy loss and the Adam optimizer.
- `test` function: Evaluates the model on the test set.
- `MyClient` class: Inherits from `fl.client.NumPyClient` and implements methods for getting/setting model parameters, training, and evaluation.

Using the Flower framework's server-client architecture, each branch participates in the model training as a client, and model aggregation is performed on the server-side. The main challenge during training was simulating a realistic federated learning environment, which was addressed by setting different random seeds and sample sizes to mimic data distribution differences across branches.

### 4. Evaluation and Results Analysis

#### Evaluation Method
The accuracy metric is used to evaluate the model's performance in identifying high-potential customers. The `fit_metrics_aggregation` function in `server.py` defines how evaluation metrics are aggregated.

#### Results Analysis
After 20 rounds of federated training, the model achieved a relatively high accuracy (approximately 0.77) on the test set. However, due to the varying data distributions across branches, the model's performance differed slightly across branches, as evident from the local evaluation results printed by the clients. This highlights the need for further model optimization to improve generalization across different data distributions.

### 5. Privacy Considerations

One of the main advantages of federated learning is data privacy protection. In this implementation, raw data never leaves the local devices; only model updates are uploaded to the server for aggregation, effectively preventing data leakage.

However, there are still potential privacy risks to consider:

- Model updates themselves may leak some information. If malicious participants forge updates, it could impact model performance.
- Communication channels need to be encrypted to prevent model updates from being intercepted during transmission.
- If the central server itself has security vulnerabilities and the model data is hijacked, there is a risk of local data being reconstructed from the model updates.

Overall, privacy protection has been maximized in this implementation, but further evaluation and enhancement of privacy safeguards would be necessary for practical applications.

### 6. Execution Process and Output Examples

1. Set up the Python environment and install the Flower framework:

```bash
pip install flwr
```

2. Open a terminal and run the server:

```bash
python server.py
```

You should see output similar to:

```bash
INFO :      Starting Flower server, config: num_rounds=20, no round_timeout
INFO :      Flower ECE: gRPC server running (20 rounds), SSL is disabled
INFO :      [INIT]
```

3. Open a new terminal and execute the client script for the first branch:

```bash
python client.py --n 1000 --simulate_seed 11 --batch_size 100
```

This defines the data for the first branch, where `n` is the number of samples, and `simulate_seed` is the random seed.

Open two more terminals and execute the client scripts for the other two branches:

```bash
python client.py --n 1200 --simulate_seed 22 --batch_size 100
```

and

```bash
python client.py --n 1500 --simulate_seed 33 --batch_size 100
```

If everything is set up correctly, you should see the training process start.

4. After the training finishes, you should see the final results in the server terminal:

```bash
INFO :      [ROUND 20]
INFO :      configure_fit: strategy sampled 3 clients (out of 3)
INFO :      aggregate_fit: received 3 results and 0 failures
INFO :      configure_evaluate: strategy sampled 3 clients (out of 3)
INFO :      aggregate_evaluate: received 3 results and 0 failures
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 20 rounds in 24.30s
INFO :      History (loss, distributed):
INFO :          ('\\tround 1: 514.3643040012669\\n'
INFO :           '\\tround 2: 112.85921004011824\\n'
INFO :           '\\tround 3: 170.58584924646325\\n'
INFO :           '\\tround 4: 161.46637746450062\\n'
……
……
```

In one of the client terminals, you should see output like:

```bash
INFO :      [RUN 0, ROUND ]
INFO :      Received: train message 8cca033b-77c0-40f8-80bd-d630dd1aaa1d
>>> Local training - Epoch 1/1 - Average Loss: 174.6097, Accuracy: 0.6687
INFO :      Sent reply
INFO :
INFO :      [RUN 0, ROUND ]
INFO :      Received: evaluate message 75a0245d-04e0-43f7-926a-9c1905d0d9d0
>>> Local testing - Average Loss: 38.2243, Accuracy: 0.7650
INFO :      Sent reply
INFO :
INFO :      [RUN 0, ROUND ]
INFO :      Received: reconnect message bf2c9e35-88e9-4428-bbe6-1d3bed78a7b4
INFO :      Disconnect and shut down
```

## Conclusion

This project successfully implemented a federated learning system to identify high-potential customers from simulated bank data distributed across 3 branches. Through a series of steps, including simulating customer data, building a federated learning system using Flower Framework, selecting and training a machine learning model, evaluating model performance, and analyzing privacy implications, it demonstrated the ability to combine privacy-preserving techniques and machine learning to solve practical problems.

Key contributions of this project include:

- Simulating diverse and realistic distributed customer data
- Implementing a federated learning system using framework like Flower
- Evaluating the model's ability to identify high-potential customers while protecting data privacy
- Analyzing the advantages and limitations of federated learning in terms of privacy protection

Future work could involve further improving the model's generalization capability, exploring more complex model architectures, and investigating additional privacy-enhancing techniques.

Overall, this project showcases how federated learning can enable valuable machine learning applications in privacy-sensitive domains like finance, providing a practical case study for the integration of data privacy and artificial intelligence.
