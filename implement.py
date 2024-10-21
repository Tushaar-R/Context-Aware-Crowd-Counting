from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms, models
import random
import torch.nn as nn
import collections
import time
from tqdm import tqdm
import math
import matplotlib.cm as CM

class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self,img_root,gt_dmap_root,gt_downsample=1,phase='train'):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        phase: train or test
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_downsample=gt_downsample
        self.phase=phase

        self.img_names=[filename for filename in os.listdir(img_root) \
                           if os.path.isfile(os.path.join(img_root,filename))]
        self.n_samples=len(self.img_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))/255# convert from [0,255] to [0,1]
        
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        gt_dmap=np.load(os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.npy')))
        
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1]#水平翻转
            gt_dmap=gt_dmap[:,::-1]#水平翻转
        
        if self.gt_downsample>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample

            img_tensor=torch.tensor(img,dtype=torch.float)
            img_tensor=transforms.functional.normalize(img_tensor,mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)

        return img_tensor,gt_dmap_tensor

def cal_mae(img_root,gt_dmap_root):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+" mae:"+str(mae/len(dataloader)))
    
def cal_rmse(img_root,gt_dmap_root):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    dataset=CrowdDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    rmse=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            rmse+=((et_dmap.data.sum()-gt_dmap.data.sum())**2).item()
            del img,gt_dmap,et_dmap
    rmse=math.sqrt(rmse/len(dataloader))

    print("model_param_path:"+" rmse:"+str(rmse))


def estimate_density_map(img_root, gt_dmap_root, index):
    '''
    Show one estimated density-map along with the original image and ground truth density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    index: the order of the test image in the test dataset.
    '''
    dataset = CrowdDataset(img_root, gt_dmap_root, 8, phase='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    model.eval()
    
    for i, (img, gt_dmap) in enumerate(dataloader):
        if i == index:
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            
            # Forward propagation to estimate density map
            et_dmap = model(img).detach()
            et_dmap = et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            gt_dmap = gt_dmap.squeeze(0).squeeze(0).cpu().numpy()
            img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            
            print("Estimated Density Map Shape:", et_dmap.shape)
            print("Ground Truth Density Map Shape:", gt_dmap.shape)

            # Create subplots
            plt.figure(figsize=(12, 4))

            # Original Image
            plt.subplot(1, 3, 1)
            plt.title('Original Image')
            plt.imshow(img)
            plt.axis('off')

            # Ground Truth Density Map
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Density Map')
            plt.imshow(gt_dmap, cmap=CM.jet)
            plt.axis('off')

            # Estimated Density Map
            plt.subplot(1, 3, 3)
            plt.title('Estimated Density Map')
            plt.imshow(et_dmap, cmap=CM.jet)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            break

class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CANNet,self).__init__()
        self.frontend_feat=[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat=[512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 1024,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.conv1_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv1_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv2_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv2_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv3_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv6_1=nn.Conv2d(512,512,kernel_size=1,bias=False)
        self.conv6_2=nn.Conv2d(512,512,kernel_size=1,bias=False)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
#            print("VGG",list(mod.state_dict().items())[0][1])#要的VGG值
            fsd=collections.OrderedDict()
            for i in range(len(self.frontend.state_dict().items())):#10个卷积*（weight，bias）=20个参数
                temp_key=list(self.frontend.state_dict().items())[i][0]
                fsd[temp_key]=list(mod.state_dict().items())[i][1]
            self.frontend.load_state_dict(fsd)
#            print("Mine",list(self.frontend.state_dict().items())[0][1])#将VGG值赋予自己网络后输出验证
#            self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]#python2.7版本
    def forward(self,x):
        fv = self.frontend(x)
        #S=1
        ave1=nn.functional.adaptive_avg_pool2d(fv,(1,1))
        ave1=self.conv1_1(ave1)
#        ave1=nn.functional.relu(ave1)
        s1=nn.functional.upsample(ave1,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
        c1=s1-fv
        w1=self.conv1_2(c1)
        w1=nn.functional.sigmoid(w1)
        #S=2
        ave2=nn.functional.adaptive_avg_pool2d(fv,(2,2))
        ave2=self.conv2_1(ave2)
#        ave2=nn.functional.relu(ave2)
        s2=nn.functional.upsample(ave2,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
        c2=s2-fv
        w2=self.conv2_2(c2)
        w2=nn.functional.sigmoid(w2)
        #S=3
        ave3=nn.functional.adaptive_avg_pool2d(fv,(3,3))
        ave3=self.conv3_1(ave3)
#        ave3=nn.functional.relu(ave3)
        s3=nn.functional.upsample(ave3,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
        c3=s3-fv
        w3=self.conv3_2(c3)
        w3=nn.functional.sigmoid(w3)
        #S=6
#        print('fv',fv.mean())
        ave6=nn.functional.adaptive_avg_pool2d(fv,(6,6))
#        print('ave6',ave6.mean())
        ave6=self.conv6_1(ave6)
#        print(ave6.mean())
#        ave6=nn.functional.relu(ave6)
        s6=nn.functional.upsample(ave6,size=(fv.shape[2],fv.shape[3]),mode='bilinear')
#        print('s6',s6.mean(),'s1',s1.mean(),'s2',s2.mean(),'s3',s3.mean())
        c6=s6-fv
#        print('c6',c6.mean())
        w6=self.conv6_2(c6)
        w6=nn.functional.sigmoid(w6)
#        print('w6',w6.mean())
        
        fi=(w1*s1+w2*s2+w3*s3+w6*s6)/(w1+w2+w3+w6+0.000000000001)
#        print('fi',fi.mean())
#        fi=fv
        x=torch.cat((fv,fi),1)
        
        x = self.backend(x)
        x = self.output_layer(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# testing
if __name__=="__main__":
    csrnet=CANNet().to('cuda')
    input_img=torch.ones((1,3,256,256)).to('cuda')
    out=csrnet(input_img)
    print(out.mean())

    #Training 
    
    train_image_root = #path of folder containing Train data images
    train_dmap_root =  #path of folder containing Train data ground truth
    test_image_root =  #path of folder containing Test data images
    test_dmap_root =   #path of folder containing Test data ground truth
    gpu_or_cpu = 'cuda'  # use cuda or cpu
    lr = 1e-7
    batch_size = 1
    momentum = 0.95
    epochs = 1000
    steps = [-1, 1, 100, 150]
    scales = [1, 1, 1, 1]
    workers = 4
    seed = time.time()
    print_freq = 30

    device = torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    model = CANNet().to(device)
    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr)
    
    train_dataset = CrowdDataset(train_image_root, train_dmap_root, gt_downsample=8, phase='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataset = CrowdDataset(test_image_root, test_dmap_root, gt_downsample=8, phase='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
        
    min_mae = 10000
    min_epoch = 0
    train_loss_list = []
    epoch_list = []
    test_error_list = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for i, (img, gt_dmap) in enumerate(tqdm(train_loader)):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # Forward propagation
            et_dmap = model(img)
            # Calculate loss
            loss = criterion(et_dmap, gt_dmap)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(train_loader))
        torch.save(model.state_dict(), './checkpoints/epoch_' + str(epoch) + ".pth")

        # Testing phase
        model.eval()
        mae = 0
        for i, (img, gt_dmap) in enumerate(tqdm(test_loader)):
            img = img.to(device)
            gt_dmap = gt_dmap.to(device)
            # Forward propagation
            et_dmap = model(img)
            mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()

        if mae / len(test_loader) < min_mae:
            min_mae = mae / len(test_loader)
            min_epoch = epoch

        test_error_list.append(mae / len(test_loader))
        print("epoch:" + str(epoch) + " error:" + str(mae / len(test_loader)) + " min_mae:" + str(min_mae) + " min_epoch:" + str(min_epoch))

        # Optionally show an image
        index = random.randint(0, len(test_loader) - 1)
        img, gt_dmap = test_dataset[index]
        # Here you could use Matplotlib to visualize if needed:
        # plt.imshow(img.squeeze(0).cpu().numpy().transpose(1, 2, 0))  # Adjust shape for visualization
        # plt.title('Image')
        # plt.show()

        # Process gt_dmap and et_dmap as needed for evaluation
        img = img.unsqueeze(0).to(device)
        gt_dmap = gt_dmap.unsqueeze(0)
        et_dmap = model(img)
        et_dmap = et_dmap.squeeze(0).detach().cpu().numpy()
        # Optionally visualize et_dmap with Matplotlib if needed

    print(time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    
    torch.backends.cudnn.enabled=False
    
    #Testing 
    img_root= test_image_root
    gt_dmap_root=test_dmap_root
    cal_mae(img_root,gt_dmap_root)
    cal_rmse(img_root,gt_dmap_root)
    estimate_density_map(img_root,gt_dmap_root,150)

