from model import Net
import torch 
from torch import nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device="cuda" if torch.cuda.is_available() else "cpu"
print(f"device used: {device}")

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])

train_data=torchvision.datasets.MNIST(root='./data',train=True,transform=transform,download=True)
test_data=torchvision.datasets.MNIST(root='./data',train=False,transform=transform,download=True)

train_dataloader=DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader=DataLoader(test_data,batch_size=32,shuffle=False)

model=Net().to(device)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=10
for epoch in range(epochs):
    model.train()
    training_loss=0.0
    for images,label in train_dataloader:
        images,label=images.to(device),label.to(device)
        optimizer.zero_grad()
        prediction=model(images)
        train_loss=loss(prediction,label)
        train_loss.backward()
        optimizer.step()
        training_loss+=train_loss.item()
        
    model.eval()
    testing_loss=0.0
    correct=0
    total=0
    with torch.no_grad():
            for images,label in test_dataloader:
                images,label=images.to(device),label.to(device)
                output=model(images)
                test_loss=loss(output,label)
                testing_loss+=test_loss.item()
                predicted=torch.argmax(output,dim=1)
                correct+=(predicted==label).sum().item()
                total+=label.size(0)
                
    avg_train_loss=training_loss/len(train_dataloader)
    avg_test_loss=testing_loss/len(test_dataloader)
    accuracy=correct*100/total
    
    print(f"{epoch}. Train loss: {avg_train_loss}  Test loss: {avg_test_loss}  Accuracy: {accuracy}")
    
torch.save(model.state_dict(),'model/digit_model.pth')
print("Model is successfully saved...")
