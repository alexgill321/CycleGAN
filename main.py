import torch
import torch.nn as nn
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import os

#%%
output_folder = path_to_folder + 'resized_surrealism'
input_folder = path_to_folder + 'zdzislaw-beksinski'

#%%
# Define the generator (ResNet-based)


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
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
dataloader_A = DataLoader(ImageDataset(path_to_folder + 'resized_realism', transform=transform), batch_size=1, shuffle=True)
dataloader_B = DataLoader(ImageDataset(path_to_folder + 'resized_surrealism', transform=transform), batch_size=1, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

n_epochs = 50
for epoch in range(n_epochs):
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

        print("Epoch: (%3d) (%5d/%5d) Loss_D_A: %.2f Loss_D_B: %.2f Loss_G: %.2f" %
              (epoch, i, len(dataloader_A), loss_D_A, loss_D_B, loss_G))

