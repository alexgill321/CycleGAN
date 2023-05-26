import torch
import torch.nn as nn
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

#%%
realism_path = os.getcwd() + '/resized_realism/'
surrealism_path = os.getcwd() + '/resized_surrealism/'

#%%
# Define the generator (ResNet-based)


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=3):
        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        # Downsampling layers
        for i in range(2):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        # Residual blocks
        mult = 2**2
        for i in range(n_blocks):
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3),
                      nn.InstanceNorm2d(ngf * mult),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3),
                      nn.InstanceNorm2d(ngf * mult)]

        # Upsampling Layers
        for i in range(2):
            mult = 2**(2-i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

#%%
# Define the discriminator (PatchGAN-based)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()

        kw = 4
        padw = int((kw-1)/2)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
#%%
# Define input and output number of channels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

t = torch.cuda.get_device_properties(0).total_memory
r = torch.cuda.memory_reserved(0)
a = torch.cuda.memory_allocated(0)
f = r-a  # free inside reserved

print("Total GPU Memory: ", t)
print("Reserved Memory: ", r)
print("Allocated Memory: ", a)
print("Free within reserved: ", f)

input_nc = 3
output_nc = 3

# Instantiate the generators
G_A2B = ResNetGenerator(input_nc, output_nc).to(device) # Generator for transforming A to B
G_B2A = ResNetGenerator(output_nc, input_nc).to(device) # Generator for transforming B to A

# Instantiate the discriminators
D_A = NLayerDiscriminator(input_nc).to(device) # Discriminator for A
D_B = NLayerDiscriminator(output_nc).to(device) # Discriminator for B
#%%
class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        super().__init__()
        self.directory = directory
        self.transform = transform
        self.file_list = os.listdir(directory)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.file_list[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
#%%

# Transforms for the input images
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataloaders for your two image folders
dataloader_A = DataLoader(ImageDataset(realism_path, transform=transform), batch_size=4, shuffle=True)
dataloader_B = DataLoader(ImageDataset(surrealism_path, transform=transform), batch_size=4, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

n_epochs = 10
avg_loss_D_A = []
avg_loss_D_B = []
avg_loss_G = []
for epoch in range(n_epochs):
    loss_D_A_list = []
    loss_D_B_list = []
    loss_G_list = []
    for i, (real_A, real_B) in enumerate(zip(dataloader_A, dataloader_B)):
        # Ensure they're on the right device
        real_A = Variable(real_A.to(device))
        real_B = Variable(real_B.to(device))

        # Generators A2B and B2A
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_cycle(G_B2A(real_B), real_B)
        loss_id_B = criterion_cycle(G_A2B(real_A), real_A)

        # GAN loss
        fake_B = G_A2B(real_A)
        pred_fake = D_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, Variable(torch.ones(pred_fake.size()).to(device)))

        fake_A = G_B2A(real_B)
        pred_fake = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, Variable(torch.ones(pred_fake.size()).to(device)))

        # Cycle loss
        recovered_A = G_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

        recovered_B = G_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

        # Total loss
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_id_A + loss_id_B
        loss_G.backward()

        optimizer_G.step()

        # Discriminator A
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = D_A(real_A)
        loss_D_real = criterion_GAN(pred_real, Variable(torch.ones(pred_real.size()).to(device)))

        # Fake loss
        pred_fake = D_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, Variable(torch.zeros(pred_fake.size()).to(device)))

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()

        # Discriminator B
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = D_B(real_B)
        loss_D_real = criterion_GAN(pred_real, Variable(torch.ones(pred_real.size()).to(device)))

        # Fake loss
        pred_fake = D_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, Variable(torch.zeros(pred_fake.size()).to(device)))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        loss_D_A_list.append(loss_D_A.item())
        loss_D_B_list.append(loss_D_B.item())
        loss_G_list.append(loss_G.item())
        print("Epoch: (%3d) (%5d/%5d) Loss_D_A: %.2f Loss_D_B: %.2f Loss_G: %.2f" %
              (epoch, i, len(dataloader_A), loss_D_A, loss_D_B, loss_G))

    # Print and save example images every 5 epochs
    if epoch % 5 == 0:
        with torch.no_grad():
            n_images = 5
            real_A_iter = iter(dataloader_A)
            real_B_iter = iter(dataloader_B)
            transformed_A = []
            transformed_B = []
            original_A = []
            original_B = []

            for _ in range(n_images):
                real_A = next(real_A_iter)
                real_B = next(real_B_iter)
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                # Generate transformed images
                fake_B = G_A2B(real_A)
                fake_A = G_B2A(real_B)

                transformed_A.append(fake_A)
                transformed_B.append(fake_B)
                original_A.append(real_A)
                original_B.append(real_B)

            # Concatenate the transformed images
            transformed_A = torch.cat(transformed_A, dim=0)
            transformed_B = torch.cat(transformed_B, dim=0)
            original_A = torch.cat(original_A, dim=0)
            original_B = torch.cat(original_B, dim=0)

            # Prepare the images for display
            transformed_A = (transformed_A + 1) / 2  # Convert from range [-1, 1] to [0, 1]
            transformed_B = (transformed_B + 1) / 2
            original_A = (original_A + 1) / 2
            original_B = (original_B + 1) / 2

            # Show the images in two columns
            fig, axs = plt.subplots(n_images, 4, figsize=(12, 3*n_images))
            fig.tight_layout()

            for i in range(n_images):
                # Display original image from column A
                axs[i, 0].imshow(original_A[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 0].axis('off')

                # Display original image from column B
                axs[i, 1].imshow(original_B[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 1].axis('off')

                # Display transformed image from column A
                axs[i, 2].imshow(transformed_A[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 2].axis('off')

                # Display transformed image from column B
                axs[i, 3].imshow(transformed_B[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 3].axis('off')

            # Save the figure
            plt.savefig('images/epoch_{}.png'.format(epoch))
            plt.close(fig)
    avg_loss_D_A.append(np.mean(loss_D_A_list))
    avg_loss_D_B.append(np.mean(loss_D_B_list))
    avg_loss_G.append(np.mean(loss_G_list))

#%%
# Generar imágenes transformadas
n_images = 5
real_A_iter = iter(dataloader_A)
real_B_iter = iter(dataloader_B)
transformed_A = []
transformed_B = []
original_A = []
original_B = []
with torch.no_grad():
    for _ in range(n_images):
        real_A = next(real_A_iter)
        real_B = next(real_B_iter)
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Generar imágenes transformadas
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        transformed_A.append(fake_A)
        transformed_B.append(fake_B)
        original_A.append(real_A)
        original_B.append(real_B)

    # Concatenar las imágenes transformadas
    transformed_A = torch.cat(transformed_A, dim=0)
    transformed_B = torch.cat(transformed_B, dim=0)
    original_A = torch.cat(original_A, dim=0)
    original_B = torch.cat(original_B, dim=0)

    # Preparar las imágenes para mostrar
    transformed_A = (transformed_A + 1) / 2  # Convertir de rango [-1, 1] a [0, 1]
    transformed_B = (transformed_B + 1) / 2
    original_A = (original_A + 1) / 2
    original_B = (original_B + 1) / 2

    # Mostrar las imágenes en dos columnas
    fig, axs = plt.subplots(n_images, 4, figsize=(12, 3*n_images))
    fig.tight_layout()

    for i in range(n_images):
        # Mostrar imagen original de la columna A
        axs[i, 0].imshow(original_A[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 0].axis('off')

        # Mostrar imagen original de la columna B
        axs[i, 1].imshow(original_B[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 1].axis('off')

        # Mostrar imagen transformada de la columna A
        axs[i, 2].imshow(transformed_A[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 2].axis('off')

        # Mostrar imagen transformada de la columna B
        axs[i, 3].imshow(transformed_B[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 3].axis('off')

    plt.show()

#%%
# Guardar los modelos
torch.save({
    'G_A2B_state_dict': G_A2B.state_dict(),
    'G_B2A_state_dict': G_B2A.state_dict(),
    'D_A_state_dict': D_A.state_dict(),
    'D_B_state_dict': D_B.state_dict(),
    'optimizer_G_state_dict': optimizer_G.state_dict(),
    'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
    'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
}, 'model_checkpoint_5Epoch_Batch6.pth')
