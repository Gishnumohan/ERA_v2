import torch 
from tqdm import tqdm

#Import pyplot from matplotlib library
import matplotlib.pyplot as plt

""" Function to mount Google Drive to your workspace """ 
def mountDrive(): 
    from google.colab import drive 
    drive.mount('/content/drive') 

""" Function to select Device """ 
def selectDevice(): 
    # Checking if we have CUDA enabled GPU or not,  
    using_cuda = torch.cuda.is_available() 
    print("Using CUDA!" if using_cuda else "Not using CUDA.") 
    # if so select "cuda" as device for processing else "cpu" 
    device = torch.device("cuda" if using_cuda else "cpu") 
    return device 

""" Function to Get Correct Prediction Count """ 
def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

""" Function to view the Training and Testing Accuracy and Loss """ 
def viewAnalysisPlt(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
