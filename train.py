# import wandb
# wandb.init(project='gan')
import torch, torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from net import Generator, Discriminator, real_loss, fake_loss

def dataloader(batch_size=1):
    train_dataset = datasets.MNIST(root='dataset', 
                                train=True, 
                                transform=transforms.ToTensor(),
                                download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size,
                                            num_workers=0)

def train(batch_size=1, latent_size=100, learning_rate=2e-3, num_epochs=100):
    dataloader = dataloader(batch_size=batch_size)
    cuda = torch.cuda.is_available()
    device = 'cuda:0' if cuda else 'cpu'
    gen_imgs = []

    G = Generator(input_size=latent_size)
    D = Discriminator()
    if cuda:
        print('Using CUDA')
        G.cuda()
        D.cuda()

    fixed_img = np.random.uniform(-1, 1, size=(batch_size, latent_size))
    fixed_img = torch.from_numpy(fixed_img).float()

    g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)
    d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)

    wandb.watch(G)
    wandb.watch(D)
    for epoch in range(num_epochs):
        D.train()
        G.train()
        for idx, ( real_images, _ ) in enumerate(tqdm(dataloader)):
            if cuda:
                real_images = real_images.cuda()

            batch_size = real_images.size(0)
            real_images = real_images * 2 - 1

            g_loss_value = 0.0
            d_loss_value = 0.0
            for phase in ['discriminator', 'generator']:
                # TRAIN DISCRIMINATOR
                if phase == 'discriminator':
                    # generate fake images from latent vector
                    latent_vector = np.random.uniform(-1, 1, size=(batch_size, latent_size))
                    latent_vector = torch.from_numpy(latent_vector).float()
                    fake_images = G(latent_vector)

                    # compute discriminator loss on real images
                    d_optimizer.zero_grad()
                    d_real = D(real_images)
                    d_real_loss = real_loss(d_real, smooth=True)

                    # compute discriminator loss in fake images
                    d_fake = D(fake_images)
                    d_fake_loss = fake_loss(d_fake)

                    # total loss, backprop, optimize and update weights
                    d_loss = d_real_loss + d_fake_loss
                    d_loss_value = d_loss.item()

                    d_loss.backward()
                    d_optimizer.step()

                # TRAIN GENERATOR
                if phase == 'generator':
                    latent_vector = np.random.uniform(-1, 1, size=(batch_size, latent_size))
                    latent_vector = torch.from_numpy(latent_vector).float()
                    fake_images = G(latent_vector)
                    
                    g_optimizer.zero_grad()
                    d_fake = D(fake_images)
                    g_loss = real_loss(d_fake)
                    g_loss_value = g_loss.item()

                    g_loss.backward()
                    g_optimizer.step()

            if idx % 100 == 0: 
                wandb.log({ 'G Loss': g_loss_value, 'D Loss': d_loss_value })
        wandb.log({ 'G Epoch Loss': g_loss_value, 'D Epoch Loss': d_loss_value }, step=epoch)
        
        # test performance
        G.eval()
        gen_img = G(fixed_img)
        gen_imgs.append(gen_img)
    
    # dump generated images
    with open('gen_imgs.pkl', 'wb') as f:
        pkl.dump(gen_imgs, f)


if __name__ == '__main__':
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    # images = torch.squeeze(images, 0).permute(1,2,0)
    # images = torch.cat((images, images, images), dim=2)
    # print(images.size())
    # plt.imshow(images)
    # plt.show()

    train()