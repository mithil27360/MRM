import torch
from torchvision import transforms
from PIL import Image
import argparse
from normal.model import SimpleCNN, LeNet5


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(model_path, image_path, model_name='simple'):
    # --- UPDATED DEVICE SELECTION ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # --------------------------------

    if model_name == 'simple':
        model = SimpleCNN().to(device)
    elif model_name == 'lenet':
        model = LeNet5().to(device)
    else:
        raise ValueError("model_name must be 'simple' or 'lenet'")


    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    print(f'Predicted Digit: {predicted_class}')
    print(f'Confidence: {confidence:.2f}%')

    print('\nClass Probabilities:')
    for i in range(10):
        prob = probabilities[0][i].item() * 100
        print(f'  {i}: {prob:.2f}%')

    return predicted_class, confidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on a digit image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--model_name', type=str, default='simple',
                        choices=['simple', 'lenet'],
                        help='Model architecture (simple or lenet)')

    args = parser.parse_args()

    predict(args.model_path, args.image_path, args.model_name)