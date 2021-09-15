#The orignal model was run on google colab and gradient paper space
#this is more or less just the code used there in the ipynb files
#apologies for not bringing it in a more object orientated format

#Define hyperparameter:
Batchsize=20
num_epochs=60
LearningRate=0.1

#import required packages
from datetime import datetime
from torch import nn
import pandas as pd
import pandas_path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as pil_image
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from PIL import Image as pil_image
import random
import sklearn.model_selection
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
from torchvision.models.video import r3d_18

torch.backends.cudnn.benchmark=True

#load pretained model and change final layer

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        model=r3d_18(pretrained=True)
        model.fc=nn.Identity()
        
        self.cnn = model
        self.fc1 = nn.Linear(512+25, 1)
                
    def forward(self, image, data, time):
        x1 = self.cnn(image)
        x2 = data
        x3 = time      
        x = torch.cat((x1,x2,x3), dim=1)
        x = F.leaky_relu(self.fc1(x))
        
        return x
        

model = MyModel()



#disable weight changes for pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

model.fc1.weight.requires_grad = True
model.fc1.bias.requires_grad = True
 
  
  
#import training and validation metadata (here just a proxy from the colab code
!unzip '/content/drive/MyDrive/HU Data/train/train.zip' -d '/content'

train_metadata = pd.read_csv('/content/drive/MyDrive/HU Data/training_set_features.csv')
train_labels = pd.read_csv('/content/drive/MyDrive/HU Data/training_set_labels.csv')

full_metadata = train_metadata.merge(train_labels, on="image_id")
full_metadata["file_name"] = (
    '/content/train/' + full_metadata.image_id.path.with_suffix(".jpg")
)


# Add a temporary column for number of images per storm
images_per_storm = full_metadata.groupby("storm_id").size().to_frame("images_per_storm")
full_metadata = full_metadata.merge(images_per_storm, how="left", on="storm_id")


#In order to get a representative validation set the a stratified split by number of images per storm is used
#create proxy classes first
storm_counts = train_metadata.groupby("storm_id").size()
df=pd.DataFrame(storm_counts,columns=["counts"]).reset_index()
df["class"]=1*(df.counts>100)+1*(df.counts>200)+1*(df.counts>300)+1*(df.counts>400)+1*(df.counts>500)+1*(df.counts>600)

sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1,  test_size=0.2,random_state=999)
split = sss.split(df, df["class"])

for a,b in split:
    IDs_train=df.iloc[a].storm_id
    IDs_val=df.iloc[b].storm_id

train=full_metadata[full_metadata["storm_id"].isin(IDs_train)].drop(
    ["images_per_storm",], axis=1
)
val=full_metadata[full_metadata["storm_id"].isin(IDs_val)].drop(
    ["images_per_storm",], axis=1
) 
  


#create Data Generator:
class DatasetWIND(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_train,transform, y_train=None):
        self.data = x_train
        self.label = y_train
        self.transform =transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        
        image_id = self.data.iloc[index]["image_id"]
        ocean = self.data.iloc[index]["ocean"]-1
        storm_id = self.data.iloc[index]["storm_id"]
        t= self.data.iloc[index]["relative_time"]
        
        all_ds = self.data[(self.data["storm_id"]==storm_id)&(self.data["relative_time"]<=t)]
        
        z=np.minimum(all_ds.shape[0],24)
        all_ds = all_ds.sort_values(["relative_time"],ascending=True)
        all_ds=all_ds[-z:]
        
        paths=list(all_ds["file_name"])
        time=list(all_ds["relative_time"].diff().fillna(all_ds["relative_time"])/(60*24*60*60))
        
        images=[]
        
        for path in paths:
            image=cv2.imread(path)
            images.append(image)
            
        while len(images)<24:
            image=cv2.imread(paths[0])
            images.insert(0,image)
            time.insert(0,time[0])
        
        time=torch.FloatTensor(time)
        
        trans=self.transform(
                             image=images[23],
                             image0=images[0],
                             image1=images[1],
                             image2=images[2],
                             image3=images[3],
                             image4=images[4],
                             image5=images[5],
                             image6=images[6],
                             image7=images[7],
                             image8=images[8],
                             image9=images[9],
                             image10=images[10],
                             image11=images[11],
                             image12=images[12],
                             image13=images[13],
                             image14=images[14],
                             image15=images[15],
                             image16=images[16],
                             image17=images[17],
                             image18=images[18],
                             image19=images[19],
                             image20=images[20],
                             image21=images[21],
                             image22=images[22]
                            )
        
        image=torch.stack(
                            [
                                trans["image0"],
                                trans["image1"],
                                trans["image2"],
                                trans["image3"],
                                trans["image4"],
                                trans["image5"],
                                trans["image6"],
                                trans["image7"],
                                trans["image8"],
                                trans["image9"],
                                trans["image10"],
                                trans["image11"],
                                trans["image12"],
                                trans["image13"],
                                trans["image14"],
                                trans["image15"],
                                trans["image16"],
                                trans["image17"],
                                trans["image18"],
                                trans["image19"],
                                trans["image20"],
                                trans["image21"],
                                trans["image22"],
                                trans["image"]
                            ]
                        ).permute(1, 0,2,3) 
        
        if self.label is not None:
            label = self.label.iloc[index]
            sample = {"image_id": image_id, "image": image, "label": label,"ocean": ocean,"time":time}
        else:
            sample = {
                "image_id": image_id,
                "image": image,
                "ocean": ocean,
                "time":time
            }
        return sample
      
      
      
      
      
#Define training and validation transformations:
names=[]
for j in range(23):
     name="image"+str(j)
     names.append(name)
targets={x:"image" for x in names}

#training transformation uses CenterCropping to focus on the eye of the hurricane as well as random bluring, cutout and rotation
transform_train=A.Compose(
            [
               
                A.transforms.CenterCrop(112,112,always_apply=True),
                A.transforms.Blur(5,p=0.1),
                A.transforms.Blur(11,p=0.2),
                A.transforms.Cutout(p=0.2),
                A.augmentations.transforms.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=180,p=0.5),
                A.transforms.Normalize (mean=(0.43216, 0.394666, 0.37645, ), std=(0.22803, 0.22145, 0.216989, ), max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
            ],
                additional_targets=targets

        )

#the validation transform just uses CenterCropping
transform_val=A.Compose(
            [
                A.transforms.CenterCrop(112,112,always_apply=True),
                A.transforms.Normalize (mean=(0.43216, 0.394666, 0.37645, ), std=(0.22803, 0.22145, 0.216989, ),max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
            ],
                additional_targets=targets
        )
      
      
#import train and validation images by used the dataloader defined above
val_set=DatasetWIND(val.drop("wind_speed",axis=1),transform_val,val["wind_speed"])
valloader=torch.utils.data.DataLoader(val_set,batch_size=Batchsize,shuffle=False,num_workers=12,pin_memory=True)

train_set=DatasetWIND(train.drop("wind_speed",axis=1),transform_train,train["wind_speed"])
trainloader=torch.utils.data.DataLoader(train_set,batch_size=Batchsize,shuffle=True,num_workers=12,pin_memory=True)
      
 


#Set up training and validation loop
criterion = nn.MSELoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LearningRate)
model.cuda()

min_val_loss=80

for epoch in range(num_epochs):  # loop over the dataset multiple times
    
    running_loss = 0.0    
    loop=tqdm(enumerate(trainloader, 0),total=len(trainloader),position=0, leave=True)

    df=pd.DataFrame(columns=["image_id","Prediction","WSPD",])
    model.train()

    for i, data in loop:
        # get the inputs; data is a list of [inputs, labels]
        inputs = data["image"].cuda()
        labels = data["label"].float().view(-1,1).cuda()
        ocean = data["ocean"].float().view(-1,1).cuda()
        time = data["time"].cuda()     
        del data
        
        # zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # forward + backward + optimize
        
        outputs = model(image=inputs,data=ocean,time=time)          
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        loop.set_description(f"Epoch[{epoch+1}/{num_epochs}] Training:")
        loop.set_postfix({"MSE":running_loss/(i+1)})
        
        
    
    running_loss = 0.0
    loop2=tqdm(enumerate(valloader, 0),total=len(valloader),position=0, leave=True)
              
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i, data in loop2:
            # get the inputs; data is a list of [inputs, labels]
            inputs_val = data["image"].cuda()
            labels_val = data["label"].float().view(-1,1).cuda()
            ocean_val = data["ocean"].float().view(-1,1).cuda()
            time_val = data["time"].cuda()
            del data

            outputs_val = model(image=inputs_val,data=ocean_val,time=time_val)
            loss_val = criterion(outputs_val, labels_val)


            d={"image_id":data["image_id"],
               "Prediction":outputs_val.cpu().detach().numpy().flatten(),
               "WSPD":labels_val.cpu().detach().numpy().flatten()}
            df2=pd.DataFrame(d)
            df=df.append(df2)

            # print statistics
            running_loss += loss_val.item()
            loop2.set_description(f"Epoch[{epoch+1}/{num_epochs}] Validation:")
            loop2.set_postfix(loss=running_loss/(i+1))
                
    
    df=df.merge(val.drop("wind_speed",axis=1),how='left',on="image_id")
    df["RMSE"]=(df["Prediction"]-df["WSPD"])**2
    X_plot = np.linspace(20, 140, 100)
    plt.plot(X_plot, X_plot, color='r')
    sns.scatterplot(data=df, x="WSPD", y="Prediction", hue="ocean",palette="deep")   
    plt.show()
    
    if epoch==0:
        for parameter in model.parameters():
            parameter.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LearningRate*0.0005)
    
    
    if running_loss/len(valloader)<min_val_loss:
        today = datetime.now()
        checkpoint = {'epoch': epoch,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}

        PATH = '/content/drive/MyDrive/HU Data/ResNet2+1D_'+today.strftime("%Y-%m-%d %H-%M-%S")+'_'+str(running_loss/len(valloader))+'.pt'
        torch.save(checkpoint, PATH)

    min_val_loss=np.min([min_val_loss,(running_loss/len(valloader))]) 
    torch.cuda.empty_cache()
