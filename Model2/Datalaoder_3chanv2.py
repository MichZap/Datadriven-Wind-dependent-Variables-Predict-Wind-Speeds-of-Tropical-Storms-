import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as pil_image
import cv2


class DatasetWIND(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_train,transforms, y_train=None):
        self.data = x_train
        self.label = y_train
        self.transform =transforms
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        image = cv2.imread(self.data.iloc[index]["file_name"], cv2.IMREAD_GRAYSCALE)
        
        
        image_id = self.data.iloc[index]["image_id"]
        time = self.data.iloc[index]["relative_time"]/(60*24*60*60)

        n=int(image_id[-3:])
        filename=self.data.iloc[index]["file_name"]
        
        image_id3=image_id[:-3]+str(int(image_id[-3:])-2).zfill(3)
        image_id4=image_id[:-3]+str(int(image_id[-3:])-1).zfill(3)
        k=len(filename)

        
        
	
        l1=len(list(self.data[self.data["image_id"]==image_id3]["relative_time"]))
        l2=len(list(self.data[self.data["image_id"]==image_id4]["relative_time"]))


        if (l2==0)&(l1==0):
          
          image3=image
          image4=image
          time3=time
          time4=time

        if (l2>0)&(l1==0):
          filename2=list(self.data[self.data["image_id"]==image_id4]["file_name"])[0]
          image4=cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)       
          image3=image4
          time4=list(self.data[self.data["image_id"]==image_id4]["relative_time"]/(60*24*60*60))[0]
          time3=time4

        if (l2>0)&(l1>0):
          filename2=list(self.data[self.data["image_id"]==image_id4]["file_name"])[0]
          filename3=list(self.data[self.data["image_id"]==image_id3]["file_name"])[0]
          image4=cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
          image3=cv2.imread(filename3, cv2.IMREAD_GRAYSCALE)
          time4=list(self.data[self.data["image_id"]==image_id4]["relative_time"]/(60*24*60*60))[0]
          time3=list(self.data[self.data["image_id"]==image_id3]["relative_time"]/(60*24*60*60))[0]
          

        
        image=cv2.merge((image3,image4,image))
        image = self.transform(image=image)

        ocean = self.data.iloc[index]["ocean"]-1
        

        if self.label is not None:
            label = self.label.iloc[index]
            sample = {"image_id": image_id, "image": image["image"], "label": label,"ocean": ocean,"time":time,"time3":time3,"time4":time4}
        else:
            sample = {
                "image_id": image_id,
                "image": image["image"],
                "ocean": ocean,
                "time":time,
                "time3":time3,
                "time4":time4,
            }
        return sample
