import torch
import pandas as pd
import numpy as np
from torch.serialization import add_safe_globals
from model import Model

add_safe_globals(['numpy._core.multiarray.scalar'])

def predict_batch(csv_path, model_path='handwriting_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = Model().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data = pd.read_csv(csv_path)
    label_mapping = checkpoint['label_mapping']
    correct_predictions = 0
    total_predictions = 0

    print("\nPredictions:")
    for i in range(6000):
        row = data.iloc[i].values
        true_label = row[0]

        pixel_values = row[1:].astype(np.float32)
        pixel_values = pixel_values / checkpoint['x_max']

        input_tensor = torch.FloatTensor(pixel_values).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, dim=1).item()

            reverse_mapping = {v: k for k, v in label_mapping.items()}
            prediction = reverse_mapping[pred_idx]

        if prediction == true_label:
            correct_predictions += 1
        total_predictions += 1

        print(f"Image {i}: True: {true_label}, Predicted: {prediction}")

    accuracy = correct_predictions / total_predictions * 100
    print(f"\nTotal Images: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    predict_batch('archive/sign_mnist_test/sign_mnist_test.csv')