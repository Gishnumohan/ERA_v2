import sys
import subprocess
import matplotlib.pyplot as plt
import torch
import numpy as np
# Overlay gradcam on top of numpy image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchmetrics import ConfusionMatrix
from torchvision import transforms
import cv2

def get_CIFAR10_musigma():
    return ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

""" Function to mount Google drive in your wokr space """ 
def mountDrive():
    from google.colab import drive
    drive.mount('/content/drive')

""" Checking if we have CUDA enabled GPU or not """
def isCudaAvailabilty():
    return torch.cuda.is_available()

""" Function to set manual seed for reproducible results """
def set_manualSeed(seed):
    # Sets the seed for PyTorch's Random Number Generator
    torch.manual_seed(seed)
    if isCudaAvailabilty():
        torch.cuda.manual_seed(seed)

""" Function to select Device """
def setDevice():
    using_cuda = isCudaAvailabilty()
    print("Using CUDA!" if using_cuda else "Not using CUDA.")
    # if so select "cuda" as device for processing else "cpu"
    device = torch.device("cuda" if using_cuda else "cpu")
    return device

""" Function to view sample data """
def show_samples(img):
    img = img / 2 + 0.5    # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

"""    Function to get the count of correct predictions.    """
def get_correct_prediction_count(prediction, label):

    return prediction.argmax(dim=1).eq(label).sum().item()


"""    Function to save the trained model along with other information to disk.    """
def save_model(epoch, model, optimizer, scheduler, batch_size, criterion, file_name):

    # print(f"Saving model from epoch {epoch}...")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "batch_size": batch_size,
            "loss": criterion,
        },
        file_name,
    )
def get_CIFAR10_musigma():
    return ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

def get_denormalized_imgs(input):
    '''
    input: normalized images, <B,C,H,W>
    '''
    mu, sigma = get_CIFAR10_musigma()
    mu = np.array(mu)
    sigma = np.array(sigma)

    # de-normalize images
    imgs = input
    npimgs = imgs.numpy()
    # de-normalize the normalized image
    npimgs = sigma[None, :, None, None] * npimgs
    npimgs = npimgs + mu[None, :, None, None]
    npimgs = np.clip(npimgs, 0, 1)
    imgs = np.transpose(npimgs, axes=(0, 2, 3, 1))
    return imgs

def get_misclassified(model, device, test_loader, misclassified, num_samples=10):
    model.eval()
    with torch.no_grad():
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(
                    dim=1, keepdim=True
                ).squeeze()
        idx = torch.where(pred != target)[0][0:num_samples]
        misclassified['data'] = data[idx].cpu()
        misclassified['pred'] = pred[idx].cpu()
        misclassified['target'] = target[idx].cpu()

def show_misclassified_imgs(misclassified, classes, nmax=10, figsize=(20,20)):
    imgs =misclassified['data'][0:nmax]
    # de-normalize the images
    imgs = get_denormalized_imgs(imgs)
    wrong_preds = misclassified['pred']
    correct_labels = misclassified['target']
    nrows = nmax
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    for i in range(nmax):
        pred_label = classes[wrong_preds[i]]
        true_label = classes[correct_labels[i]]
        axes[i].imshow(imgs[i])
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
        axes[i].set_title(f"Pred:{pred_label}, True:{true_label}")
    plt.show()

def get_gradcam_img(model, target_layer, input_tensor, imgs, preds):
    '''
    model: your trained model obj
    target_layer: layer in the model where you want to esimate Grad-CAM
    input_tensor: input to model (usually normalized image) <1,C,H,W>
    imgs: original de-normalized images <B,C,H,W>
    preds: can be preds or ground truth (label for which Grad CAM needs to be computed)
    '''
    '''
        model: your trained model obj
        target_layer: layer in the model where you want to esimate Grad-CAM
        input_tensor: input to model (usually normalized image) <1,C,H,W>
        imgs: original de-normalized images <B,C,H,W>
        preds: can be preds or ground truth (label for which Grad CAM needs to be computed)
        '''
    targets = [ClassifierOutputTarget(pr) for pr in preds]
    target_layers = [model.layer3[-1]]
    cam_images = np.ones(imgs.shape).astype(np.uint8)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        print(grayscale_cams.shape)
        for i in range(imgs.shape[0]):
            cam_images[i] = np.uint8(show_cam_on_image(imgs[i], grayscale_cams[i], use_rgb=True))

    return grayscale_cams, cam_images

def show_gradcam_plots(grayscale_cams, cam_image, original_imgs, classes, \
                       preds, labels, resize=(32,32), figsize=(20,20)):
    '''
    grayscale_cams: output from get_gradcam_img(), grayscale gradcam image <B,C,H,W>
    cam_image: output from get_gradcam_img(), grayscale gradcam overlaid on input image (de-normalized) <B,C,H,W>
    original_imgs: de-normalized <B,C,H,W>
    classes: for classifier
    preds: from classifier
    label: ground truth
    '''
    nrows = grayscale_cams.shape[0]
    fig, axes = plt.subplots(nrows, 1, figsize=figsize)
    for i in range(grayscale_cams.shape[0]):
        cam = np.uint8(255 * grayscale_cams[i, :, :])
        cam = cv2.merge([cam, cam, cam])  # grayscale
        cam_image_ = cam_image[i]  # overlaid (gradCAM grayscale + input image)
        img_ = np.uint8(255 * original_imgs[i])  # original image
        # rescale for visibility
        cam = cv2.resize(cam, resize, interpolation=cv2.INTER_CUBIC)
        cam_image_ = cv2.resize(cam_image_, resize, interpolation=cv2.INTER_CUBIC)
        image_ = cv2.resize(img_, resize, interpolation=cv2.INTER_CUBIC)
        single_img_ = np.hstack((image_, cam, cam_image_))
        # set the labels
        pred_label = classes[preds[i]]
        true_label = classes[labels[i]]
        axes[i].imshow(single_img_)
        axes[i].set_xticks([], [])
        axes[i].set_yticks([], [])
        axes[i].set_title(f"Pred:{pred_label}, True:{true_label}")
    plt.show()