# load packages

import os
import numpy as np
import time
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# define CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

batch_size = 128

# load the dataset
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)


num_images_train = 50000
num_images_test = 10000
n_classes=10


#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln1 = nn.LayerNorm(torch.Size([32, 32]), eps=1e-05, elementwise_affine=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln2 = nn.LayerNorm(torch.Size([16, 16]), eps=1e-05, elementwise_affine=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln3 = nn.LayerNorm(torch.Size([16, 16]), eps=1e-05, elementwise_affine=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln4 = nn.LayerNorm(torch.Size([8, 8]), eps=1e-05, elementwise_affine=True)
        self.conv5 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln5 = nn.LayerNorm(torch.Size([8, 8]), eps=1e-05, elementwise_affine=True)
        self.conv6 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln6 = nn.LayerNorm(torch.Size([8, 8]), eps=1e-05, elementwise_affine=True)
        self.conv7 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.ln7 = nn.LayerNorm(torch.Size([8, 8]), eps=1e-05, elementwise_affine=True)
        self.conv8 = nn.Conv2d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.ln8 = nn.LayerNorm(torch.Size([4, 4]), eps=1e-05, elementwise_affine=True)
        self.pool = nn.MaxPool2d(4, stride = 4)      
        self.fc1 = nn.Linear(128, 1)
        self.fc10 = nn.Linear(128, 10)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)


    def forward(self, x, extract_features=0):
        
        x = self.lrelu(self.ln1(self.conv1(x)))
        x = self.lrelu(self.ln2(self.conv2(x)))
        x = self.lrelu(self.ln3(self.conv3(x)))
        x = self.lrelu(self.ln4(self.conv4(x)))  
        
        if(extract_features==4):
            x = F.max_pool2d(x,8,8)
            x = x.view(-1, 128)
            return x,x
        
        x = self.lrelu(self.ln5(self.conv5(x)))
        x = self.lrelu(self.ln6(self.conv6(x)))
        x = self.lrelu(self.ln7(self.conv7(x)))
        x = self.lrelu(self.ln8(self.conv8(x)))
        
        if(extract_features==8):
            x = F.max_pool2d(x,4,4)
            x = x.view(-1, 128)
            return x,x
        
        x = self.pool(x)
        x = x.view(-1, 128)
        x1 = self.fc1(x)
        x10 = self.fc10(x)
        
        return x1, x10


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 128*4*4)
        self.conv1 = nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.ConvTranspose2d(128, 128, kernel_size = 4, stride = 2, padding = 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 3, kernel_size = 3, stride = 1, padding = 1)  
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1,128,4,4)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = torch.tanh(self.conv8(x))
        
        return x
    
    
def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.to(device)
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    
    return gradient_penalty


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


def main():
    
    #Baseline model: Dicriminator
    model =  discriminator()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    #ACGAN Discriminator
    aD =  discriminator()
    aD = aD.to(device)
    
    #ACGAN Generator
    aG = generator()
    aG = aG.to(device)
    
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
    
    criterion = nn.CrossEntropyLoss()
        
    ##########################################################################
    ### Part 1-1 Training the Discriminator WITHOUT the Generator
    ##########################################################################

    print('Start Training the Discriminator WITHOUT the Generator')
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones = [40,70],gamma=0.1)
#    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96)
    
    # train the network
    num_epoch = 100
    epoch_acc_dict = []
    accuracy_test = 0.0
    
    time01 = time.time()
    
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        
        model.train()
        
        time1 = time.time()
        
        train_accu = []
        
        scheduler.step()
        
        for param_group in optimizer.param_groups:
            print("Epoch # %3d,  Learning Rate %10.6f" % ( epoch, param_group['lr'] ) )
                  
        # progress bar
        bar_width_0 = 50
        bar_dt = round(num_images_train / (batch_size * bar_width_0))
        bar_width = num_images_train // (batch_size * bar_dt) + 1
        sys.stdout.write("[%s]" % (" " * bar_width))
        sys.stdout.flush()
        sys.stdout.write('\b' * (bar_width+1))
                  
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
            
            # progress bar
            if batch_idx % bar_dt == 0:
                sys.stdout.write("-")
                sys.stdout.flush()

            if(Y_train_batch.shape[0] < batch_size):
                continue
                
            X_train_batch = X_train_batch.to(device)
            Y_train_batch = Y_train_batch.to(device)
            
            _, output = model(X_train_batch)
            loss = criterion(output, Y_train_batch)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            # accuracy on the training set.
            prediction = output.data.max(1)[1] # first column has actual prob.
            accuracy = ( float( prediction.eq(Y_train_batch.data).sum() ) /float(batch_size)
            )*100.0
            train_accu.append(accuracy)
            
        # progress bar
        sys.stdout.write("\n")
        
        accuracy_epoch = np.mean(train_accu)
        epoch_acc_dict.append(accuracy_epoch)
        
        time2 = time.time()
        time_epoch = time2-time1
        
        print("Epoch # %3d,  Accuracy %6.2f %%, Training Time %8.2f (s)" % ( epoch, accuracy_epoch, time_epoch ) )
              
        # save model every 5 epochs
        if epoch % 5 == 4:
            model_filepath = "cifar10_" + str(epoch) + ".ckpt"
            model_state = {'epoch': epoch,
                           'epoch_acc': epoch_acc_dict,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()}
            torch.save(model_state, model_filepath)
            
            state = torch.load(model_filepath)
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])
            epoch_acc_dict = state['epoch_acc']
            
            # test accuracy of whole data
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
        
                # progress bar
                bar_width_0 = 50
                bar_dt = round(num_images_test / (batch_size * bar_width_0))
                bar_width = num_images_test // (batch_size * bar_dt) + 1
                sys.stdout.write("[%s]" % (" " * bar_width))
                sys.stdout.flush()
                sys.stdout.write('\b' * (bar_width+1))
        
                for i, data in enumerate(testloader, 0):
        
                    # progress bar
                    if i % bar_dt == 0:
                        sys.stdout.write("-")
                        sys.stdout.flush()
            
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    _, outputs = model.forward(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        
                # progress bar
                sys.stdout.write("\n")
                accuracy_test = 100 * correct / total
    
            print('Epoch # %3d: Accuracy of the network on the test images: %6.2f %%' 
                  % (epoch, accuracy_test))
            
            
        if accuracy_test > 85.0 or epoch == num_epoch - 1:
            print("Target Accuracy Reached at Epoch # %3d" % ( epoch ) )
            #save the best/final model
            model_filepath = "cifar10_best" + ".ckpt"
            model_state = {'epoch': epoch,
                           'epoch_acc': epoch_acc_dict,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()}
            torch.save(model_state, model_filepath)
            break
    
    time02 = time.time()
    time_train = time02-time01
    
    # save model
    torch.save(model,'cifar10.model')
    
    print('Finished Training')
    print("Total Training Time: %10.2f (s)" % time_train )

    # prediction on test data
    print('Apply Trained Model on Test Data')
    
#    # Load the best/final model
#    model_filepath = "cifar10_best" + ".ckpt"
#    state = torch.load(model_filepath)
#    model.load_state_dict(state['state_dict'])
#    optimizer.load_state_dict(state['optimizer'])
#    epoch_acc_dict = state['epoch_acc']
    
    model = torch.load('cifar10.model')
    
    # test accuracy of whole data
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        
        # progress bar
        bar_width_0 = 50
        bar_dt = round(num_images_test / (batch_size * bar_width_0))
        bar_width = num_images_test // (batch_size * bar_dt) + 1
        sys.stdout.write("[%s]" % (" " * bar_width))
        sys.stdout.flush()
        sys.stdout.write('\b' * (bar_width+1))
        
        for i, data in enumerate(testloader, 0):
            
            # progress bar
            if i % bar_dt == 0:
                sys.stdout.write("-")
                sys.stdout.flush()
            
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _, outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # progress bar
        sys.stdout.write("\n")
    
    print('Accuracy of the network on the test images: %6.2f %%' % (
        100 * correct / total))
    
    ##########################################################################
    ### Part 1-2 Training the Discriminator wWITH the Generator
    ##########################################################################

    print('Start Training the Discriminator WITH the Generator')
    
    # random batch of noise for the generator
    n_z = 100
    np.random.seed(352)
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0,1,(100,n_z))
    label_onehot = np.zeros((100,n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = Variable(save_noise).to(device)
    
    
    start_time = time.time()
    num_epochs = 50
    gen_train = 1 # train the generator at every 1 iteration
    
    # before epoch training loop starts
    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []
    
    #ACGAN Discriminator
    aD =  torch.load('discriminator.model')
    aD = aD.to(device)
    
    #ACGAN Generator
    aG = torch.load('generator.model')
    aG = aG.to(device)
    
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))
    
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(0,num_epochs):

        aG.train()
        aD.train()
        
        time01 = time.time()
        
        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
            
            if(Y_train_batch.shape[0] < batch_size):
                continue
            
            # train G
            if((batch_idx%gen_train)==0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()
                
                label = np.random.randint(0,n_classes,batch_size)
                noise = np.random.normal(0,1,(batch_size,n_z))
                label_onehot = np.zeros((batch_size,n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = Variable(noise).to(device)
                fake_label = Variable(torch.from_numpy(label)).to(device)
                
                fake_data = aG(noise)
                gen_source, gen_class  = aD(fake_data)
                
                gen_source = gen_source.mean()
                gen_class = criterion(gen_class, fake_label)
                
                gen_cost = -gen_source + gen_class
                gen_cost.backward()
                
                optimizer_g.step()
                
            
            # train D
            for p in aD.parameters():
                p.requires_grad_(True)
                
            aD.zero_grad()
            
            # train discriminator with input from generator
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).to(device)
            fake_label = Variable(torch.from_numpy(label)).to(device)
            with torch.no_grad():
                fake_data = aG(noise)
            
            disc_fake_source, disc_fake_class = aD(fake_data)
            
            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)
            
            # train discriminator with input from the discriminator
            real_data = Variable(X_train_batch).to(device)
            real_label = Variable(Y_train_batch).to(device)
            
            disc_real_source, disc_real_class = aD(real_data)
            
            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0
            
            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)
            
            gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)
            
            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()
            
            optimizer_d.step()
            
            # loss
            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            
            # within the training loop
            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx%50)==0):
                print('---\nEpoch #%3d, Batch #%5d, Accuracy %.2f %%, Accumulated Time %.2f (s)' 
                      % (epoch, batch_idx, np.mean(acc1), time.time()-time01))
                print('Loss1 = %.2f, Loss2 = %.2f, Loss3 = %.2f, Loss4 = %.2f, Loss5 = %.2f' 
                      % (np.mean(loss1), np.mean(loss2), np.mean(loss3), 
                         np.mean(loss4), np.mean(loss5)))
                sys.stdout.flush()
                
        time02 = time.time()
        time_epoch = (time02-time01)/3600.0
        print("=====\nEpoch # %3d,  Accuracy %.2f %%, Training Time %.2f (hrs)" % ( epoch, np.mean(acc1), time_epoch ) )
        
        # Test the model
        aD.eval()
        with torch.no_grad():
            test_accu = []
            
            # progress bar
            bar_width_0 = 50
            bar_dt = round(num_images_test / (batch_size * bar_width_0))
            bar_width = num_images_test // (batch_size * bar_dt) + 1
            sys.stdout.write("[%s]" % (" " * bar_width))
            sys.stdout.flush()
            sys.stdout.write('\b' * (bar_width+1))
            
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
                
                # progress bar
                if batch_idx % bar_dt == 0:
                    sys.stdout.write("-")
                    sys.stdout.flush()
                
                X_test_batch, Y_test_batch= Variable(X_test_batch).to(device),Variable(Y_test_batch).to(device)
                    
                with torch.no_grad():
                    _, output = aD(X_test_batch)

                prediction = output.data.max(1)[1] # first column has actual prob.
                accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
                test_accu.append(accuracy)
                accuracy_test = np.mean(test_accu)
                
            # progress bar
            sys.stdout.write("\n")
            
            print('=====\nTesting Accuracy %.2f %%, Testing Time %.2f (s)' % (accuracy_test, (time.time()-time02)))
            
            
        ### save output
        with torch.no_grad():
            aG.eval()
            samples = aG(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0,2,3,1)
            aG.train()
            
        fig = plot(samples)
        plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)
        
        if(((epoch+1)%1)==0):
            torch.save(aG,'tempG.model')
            torch.save(aD,'tempD.model')
    
    time_total =  (time.time()-start_time)/3600.0
    print('#####\nTotal Training/Testing Time %.2f (hrs)' % time_total)
    
    # save the final models D and G             
    torch.save(aG,'generator.model')
    torch.save(aD,'discriminator.model')

    ##########################################################################
    ### Part 2-1 Perturb Real Images
    ##########################################################################
    
    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    testloader = enumerate(testloader)
    
    model = torch.load('cifar10.model')
    model.cuda()
    model.eval()
    
    batch_idx, (X_batch, Y_batch) = testloader.__next__()
    X_batch = Variable(X_batch,requires_grad=True).to(device)
    Y_batch_alternate = (Y_batch + 1)%10
    Y_batch_alternate = Variable(Y_batch_alternate).to(device)
    Y_batch = Variable(Y_batch).to(device)
    
    ## save real images
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
            
    fig = plot(samples[0:100])
    plt.savefig('visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)
    
    _, output = model(X_batch)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)
    
    ## slightly jitter all input images
    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                                    grad_outputs=torch.ones(loss.size()).cuda(),
                                    create_graph=True, retain_graph=False, only_inputs=True)[0]
    
    # save gradient jitter
    gradient_image = gradients.data.cpu().numpy()
    gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
    gradient_image = gradient_image.transpose(0,2,3,1)
    fig = plot(gradient_image[0:100])
    plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
    plt.close(fig)
    
    
    # jitter input image
    gradients[gradients>0.0] = 1.0
    gradients[gradients<0.0] = -1.0
    
    gain = 8.0
    X_batch_modified = X_batch - gain*0.007843137*gradients
    X_batch_modified[X_batch_modified>1.0] = 1.0
    X_batch_modified[X_batch_modified<-1.0] = -1.0
    
    ## evaluate new fake images
    _, output = model(X_batch_modified)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)
    
    ## save fake images
    samples = X_batch_modified.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
    plt.close(fig)
    
    ##########################################################################
    ### Part 2-2 Synthetic Images Maximizing Classification Output
    ##########################################################################
    
    # discriminator trained without the generator
    model1 = torch.load('cifar10.model')
    model1.cuda()
    model1.eval()
    
    X = X_batch.mean(dim=0)
    X = X.repeat(10,1,1,1)

    Y = torch.arange(10).type(torch.int64)
    Y = Variable(Y).to(device)
    
    lr = 0.1
    weight_decay = 0.001
    
    for i in range(200):
        _, output = model1(X)
        
        loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples)
    plt.savefig('visualization/max_class_woGenerator.png', bbox_inches='tight')
    plt.close(fig)
    
    # discriminator trained without the generator
    model2 = torch.load('discriminator.model')
    model2.cuda()
    model2.eval()
    
    
    X = X_batch.mean(dim=0)
    X = X.repeat(10,1,1,1)

    Y = torch.arange(10).type(torch.int64)
    Y = Variable(Y).to(device)
    
    for i in range(200):
        _, output = model2(X)
        
        loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples)
    plt.savefig('visualization/max_class_wGenerator.png', bbox_inches='tight')
    plt.close(fig)
    
    
    ##########################################################################
    ### Part 2-3 Synthetic Features Maximizing Features at Various Layers
    ##########################################################################
    
    # discriminator trained WITHOUT the generator
    # layer 4
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):

        _, output = model1(X,extract_features=4)
        
        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features_woGenerator_layer4.png', bbox_inches='tight')
    plt.close(fig)
    
    # layer 8
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, output = model1(X,extract_features=8)
        
        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features_woGenerator_layer8.png', bbox_inches='tight')
    plt.close(fig)
    
    # discriminator trained WITH the generator
    # layer 4
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):

        _, output = model2(X,extract_features=4)
        
        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features_wGenerator_layer4.png', bbox_inches='tight')
    plt.close(fig)
    
    # layer 8
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    
    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()
    
    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        _, output = model2(X,extract_features=8)
        
        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]
        
        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)
        
        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0
        
    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)
    
    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features_wGenerator_layer8.png', bbox_inches='tight')
    plt.close(fig)
    
if __name__=='__main__':
    main()
