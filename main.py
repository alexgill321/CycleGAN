import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

#%%
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
directory = os.getcwd()
images_path = directory + '/image_data/resized_imgs/'
surrealism_path = directory + '/image_data/resized_surrealism_sel/'

#%%
# Define the generator (ResNet-based)


class ResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6):
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
    def __init__(self, directory_A, directory_B, transform=None):
        super().__init__()
        self.directory_A = directory_A
        self.directory_B = directory_B
        self.transform = transform
        self.file_list_A = os.listdir(directory_A)
        self.file_list_B = os.listdir(directory_B)
        self.max_length = max(len(self.file_list_A), len(self.file_list_B))

    def __len__(self):
        return self.max_length

    def __getitem__(self, idx):
        idx_A = idx % len(self.file_list_A)
        idx_B = idx % len(self.file_list_B)

        img_path_A = os.path.join(self.directory_A, self.file_list_A[idx_A])
        img_path_B = os.path.join(self.directory_B, self.file_list_B[idx_B])

        img_A = Image.open(img_path_A).convert('RGB')
        img_B = Image.open(img_path_B).convert('RGB')

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

#%%


class ImageBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            # the current capacity of the buffer
            self.curr_cap = 0
            # initialize buffer as empty list
            self.buffer = []

    def __call__(self, imgs):
        # the buffer is not used
        if self.buffer_size == 0:
            return imgs

        return_imgs = []
        for img in imgs:
            img = img.unsqueeze(dim=0)

            # fill buffer to maximum capacity
            if self.curr_cap < self.buffer_size:
                self.curr_cap += 1
                self.buffer.append(img)
                return_imgs.append(img)
            else:
                p = np.random.uniform(low=0., high=1.)

                # swap images between input and buffer with probability 0.5
                if p > 0.5:
                    idx = np.random.randint(low=0, high=self.buffer_size)
                    tmp = self.buffer[idx].clone()
                    self.buffer[idx] = img
                    return_imgs.append(tmp)
                else:
                    return_imgs.append(img)
        return torch.cat(return_imgs, dim=0)

#%%
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Dataloaders for your two image folders
dataloader = DataLoader(
    ImageDataset(images_path, surrealism_path, transform=transform),
    batch_size=2,
    shuffle=True
)

val_iterator = iter(dataloader)
#%%
n_images = 5
validation_a = []
validation_b = []
validation_a_use = []
validation_b_use = []
for _ in range(n_images):
    val_A, val_B = next(val_iterator)
    validation_a_use.append(val_A)
    validation_b_use.append(val_B)
# Concatenate the images
validation_a = torch.cat(validation_a_use, dim=0)
validation_b = torch.cat(validation_b_use, dim=0)

# Prepare the images for display
validation_a = (validation_a + 1) / 2  # Convert from range [-1, 1] to [0, 1]
validation_b = (validation_b + 1) / 2


# Show the images in two columns
fig, axs = plt.subplots(n_images, 2, figsize=(6, 3*n_images))
fig.tight_layout()

for i in range(n_images):
    # Display original image from column A
    axs[i, 0].imshow(validation_a[i].permute(1, 2, 0).detach().cpu().numpy())
    axs[i, 0].axis('off')

    # Display original image from column B
    axs[i, 1].imshow(validation_b[i].permute(1, 2, 0).detach().cpu().numpy())
    axs[i, 1].axis('off')
plt.show()
#%%


# Optimizers
optimizer_G = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

buffer_fake_A = ImageBuffer(100)
buffer_fake_B = ImageBuffer(100)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

n_epochs = 200
avg_loss_D_A = []
avg_loss_D_B = []
avg_loss_G = []
for epoch in range(n_epochs):
    loss_D_A_list = []
    loss_D_B_list = []
    loss_G_list = []
    start_time = time.time()
    for i, (real_A, real_B) in enumerate(dataloader):
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
        pred_fake = D_A(buffer_fake_A(fake_A.detach()))
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
        pred_fake = D_B(buffer_fake_B(fake_B.detach()))
        loss_D_fake = criterion_GAN(pred_fake, Variable(torch.zeros(pred_fake.size()).to(device)))

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake)*0.5
        loss_D_B.backward()

        optimizer_D_B.step()

        loss_D_A_list.append(loss_D_A.item())
        loss_D_B_list.append(loss_D_B.item())
        loss_G_list.append(loss_G.item())
        epoch_time = time.time() - start_time
        print("Epoch: (%3d) (%5d/%5d) Loss_D_A: %.2f Loss_D_B: %.2f Loss_G: %.2f Time: %.2f" %
              (epoch, i, len(dataloader), loss_D_A, loss_D_B, loss_G, epoch_time))

    # Print and save example images every 5 epochs
    if epoch % 1 == 0:
        with torch.no_grad():
            n_images = 5
            transformed_A = []
            transformed_B = []
            cycled_A = []
            cycled_B = []
            original_A = []
            original_B = []
            for i in range(n_images):
                real_A = validation_a_use[i]
                real_B = validation_b_use[i]
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                # Generate transformed images
                fake_B = G_A2B(real_A)
                fake_A = G_B2A(real_B)

                # Cycle back the images
                cyc_A = G_B2A(fake_B)
                cyc_B = G_A2B(fake_A)

                transformed_A.append(fake_A)
                transformed_B.append(fake_B)
                cycled_A.append(cyc_A)
                cycled_B.append(cyc_B)
                original_A.append(real_A)
                original_B.append(real_B)

            # Concatenate the images
            transformed_A = torch.cat(transformed_A, dim=0)
            transformed_B = torch.cat(transformed_B, dim=0)
            cycled_A = torch.cat(cycled_A, dim=0)
            cycled_B = torch.cat(cycled_B, dim=0)
            original_A = torch.cat(original_A, dim=0)
            original_B = torch.cat(original_B, dim=0)

            # Prepare the images for display
            transformed_A = (transformed_A + 1) / 2  # Convert from range [-1, 1] to [0, 1]
            transformed_B = (transformed_B + 1) / 2
            cycled_A = (cycled_A + 1) / 2
            cycled_B = (cycled_B + 1) / 2
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
                axs[i, 2].imshow(transformed_B[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 2].axis('off')

                # Display transformed image from column B
                axs[i, 3].imshow(transformed_A[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 3].axis('off')
            if not os.path.exists(directory + '/images'):
                os.makedirs(directory + '/images')
            img_file = directory + '/images/full_epoch_{}.png'.format(epoch)
            # Save the figure
            plt.savefig(img_file)

            plt.show()
            plt.close(fig)

            fig, axs = plt.subplots(n_images, 3, figsize=(12, 3*n_images))
            fig.tight_layout()

            for i in range(n_images):
                # Display original image from column A
                axs[i, 0].imshow(original_A[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 0].axis('off')

                # Display transformed image from column A
                axs[i, 1].imshow(transformed_B[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 1].axis('off')

                # Display cycled image from column A
                axs[i, 2].imshow(cycled_A[i].permute(1, 2, 0).detach().cpu().numpy())
                axs[i, 2].axis('off')
            img_file = directory + '/images/full_cycled_epoch_{}.png'.format(epoch)
            if not os.path.exists(directory + '/images'):
                os.makedirs(directory + '/images')
            plt.savefig(img_file)

            plt.show()

            plt.close(fig)

            data_directory = directory + '/data/losses'
            avg_loss_D_A_dir = data_directory + '/D_A_loss/'
            avg_loss_D_B_dir = data_directory + '/D_B_loss/'
            avg_loss_G_dir = data_directory + '/G_loss/'

            if not os.path.exists(avg_loss_D_A_dir):
                os.makedirs(avg_loss_D_A_dir)
            if not os.path.exists(avg_loss_D_B_dir):
                os.makedirs(avg_loss_D_B_dir)
            if not os.path.exists(avg_loss_G_dir):
                os.makedirs(avg_loss_G_dir)

            avg_loss_D_A_file = avg_loss_D_A_dir + 'epoch_{}.pkl'.format(epoch)
            avg_loss_D_B_file = avg_loss_D_B_dir + 'epoch_{}.pkl'.format(epoch)
            avg_loss_G_file = avg_loss_G_dir + 'epoch_{}.pkl'.format(epoch)

            with open(avg_loss_D_A_file, 'wb') as f:
                pickle.dump(avg_loss_D_A, f)
            with open(avg_loss_D_B_file, 'wb') as f:
                pickle.dump(avg_loss_D_B, f)
            with open(avg_loss_G_file, 'wb') as f:
                pickle.dump(avg_loss_G, f)
    avg_loss_D_A.append(np.mean(loss_D_A_list))
    avg_loss_D_B.append(np.mean(loss_D_B_list))
    avg_loss_G.append(np.mean(loss_G_list))

#%%
# Generar imágenes transformadas
n_images = 5
iterator = iter(dataloader)
transformed_A = []
transformed_B = []
original_A = []
original_B = []
with torch.no_grad():
    for _ in range(n_images):
        real_A, real_B = next(iterator)
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
n_images = 5
iterator = iter(dataloader)
transformed_A = []
transformed_B = []
cycled_A = []
cycled_B = []
original_A = []
original_B = []

with torch.no_grad():
    for _ in range(n_images):
        real_A, real_B = next(iterator)
        real_A = real_A.to(device)
        real_B = real_B.to(device)

        # Generate transformed images
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        # Cycle back the images
        cycled_A = G_B2A(fake_B)
        cycled_B = G_A2B(fake_A)

        transformed_A.append(fake_A)
        transformed_B.append(fake_B)
        cycled_A.append(cycled_A)
        cycled_B.append(cycled_B)
        original_A.append(real_A)
        original_B.append(real_B)

    # Concatenate the images
    transformed_A = torch.cat(transformed_A, dim=0)
    transformed_B = torch.cat(transformed_B, dim=0)
    cycled_A = torch.cat(cycled_A, dim=0)
    cycled_B = torch.cat(cycled_B, dim=0)
    original_A = torch.cat(original_A, dim=0)
    original_B = torch.cat(original_B, dim=0)

    # Prepare the images for display
    transformed_A = (transformed_A + 1) / 2  # Convert from range [-1, 1] to [0, 1]
    transformed_B = (transformed_B + 1) / 2
    cycled_A = (cycled_A + 1) / 2
    cycled_B = (cycled_B + 1) / 2
    original_A = (original_A + 1) / 2
    original_B = (original_B + 1) / 2

    # Display the images in three columns
    fig, axs = plt.subplots(n_images, 6, figsize=(18, 3*n_images))
    fig.tight_layout()

    for i in range(n_images):
        # Display original image from column A
        axs[i, 0].imshow(original_A[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 0].axis('off')

        # Display transformed image from column A
        axs[i, 1].imshow(transformed_A[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 1].axis('off')

        # Display cycled image from column A
        axs[i, 2].imshow(cycled_A[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 2].axis('off')

        # Display original image from column B
        axs[i, 3].imshow(original_B[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 3].axis('off')

        # Display transformed image from column B
        axs[i, 4].imshow(transformed_B[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 4].axis('off')

        # Display cycled image from column B
        axs[i, 5].imshow(cycled_B[i].permute(1, 2, 0).detach().cpu().numpy())
        axs[i, 5].axis('off')

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
