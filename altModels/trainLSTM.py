import torch
from torch.utils.data import DataLoader

from model.builder import create_preprocessing_model
from model.dunes import DataLoaderDUNES, LSTMModel

# LSTMModel and DataLoaderDUNES are imported

# Parameters for the LSTM model
input_size = 512  # Should be equal to self.vector_size in DataLoaderDUNES
hidden_size = 256
num_layers = 2
num_outputs = 3  # Adjust if different outputs are needed

# Initialize the LSTM model
lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_outputs=num_outputs)

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lstm_model.to(device)

# Example data loading setup
data = [...]  # Your dataset
preprocessing_model = ...  # Your preprocessing model with feature_size defined
dataset = DataLoaderDUNES(data, preprocessing_model)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example training loop
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Assuming regression type outputs

lstm_model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    for seq_features, target_features in dataloader:
        seq_features, target_features = seq_features.to(device), target_features.to(device)

        # Forward pass
        outputs = lstm_model(seq_features)
        loss = criterion(outputs, target_features)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
