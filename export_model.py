import torch
import torchvision
import torchvision.transforms as transforms
from model import SimpleCNN


def main():
    # ladda CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=2
    )

    # träna modellen några epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Tränar modellen...")
    for epoch in range(3):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"  Epoch {epoch+1}/3, Loss: {running_loss / len(train_loader):.4f}")

    # spara som ONNX
    model.eval()
    model = model.to("cpu")
    dummy = torch.randn(1, 3, 32, 32)

    torch.onnx.export(model, dummy, "model.onnx",
                      input_names=["image"], output_names=["output"])
    print("Klar! Sparad som model.onnx")


if __name__ == "__main__":
    main()
