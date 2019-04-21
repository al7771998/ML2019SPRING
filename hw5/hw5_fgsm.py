import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from torchvision.models import vgg16
from torchvision.models import vgg19
from torchvision.models import resnet50
from torchvision.models import resnet101
from torchvision.models import densenet121
from torchvision.models import densenet169
from PIL import Image
from scipy.misc import imsave
import pandas as pd
import numpy as np
import sys
Labels = pd.read_csv('labels.csv')
Labels = Labels.iloc[:,3].values
eps = 2e-2 #這個差距不大
model = resnet50(pretrained=True).cuda()
loss = nn.CrossEntropyLoss()
model.eval();
L_inf = 0.0
image_num = 200
succ = 0.0 
imtotensor = T.ToTensor()
tensortoim = T.ToPILImage()
seed = 21666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#torch.backends.cudnn.enabled = False 
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
for i in range(image_num):
    img = Image.open(sys.argv[1]+'/%03d.png'%(i)).convert('RGB')
    img_in = imtotensor(img)
    img_in = img_in.unsqueeze(0).cuda()
    
    x = Variable(img_in, requires_grad = True)
    y_true = Labels[i]   
    y = Variable(torch.LongTensor([y_true]).cuda(), requires_grad=False)
    zero_gradients(x)
    y_pred = model(x)
    gradient = loss(y_pred, y)
    gradient.backward(retain_graph=True)
    x.data = x.data + eps * torch.sign(x.grad.data)
    x.data = torch.clamp(x.data, 0., 1.)
    img_adv = x.data.cpu()
    img_out = tensortoim(img_adv[0])    
    img_out = np.asarray(img_out)
    saving = Image.fromarray(img_out)
    saving.save(sys.argv[2]+'/%03d.png'%(i))
    img_out = img_out.astype(np.int32)
    img = img_in.cpu()
    img = np.asarray(img)
    img *= 255
    img = np.rollaxis(img[0,:,:,:],0,3)
    img = img.astype(np.int32)
    L_inf += np.amax(np.abs(img_out - img))
    print(L_inf / (i+1))
    y_after, y_before = model(Variable(img_adv, requires_grad=True).cuda()).data.max(1)[1].cpu().numpy()[0]\
    ,model(Variable(img_in.cpu(), requires_grad=True).cuda()).data.max(1)[1].cpu().numpy()[0]
    print('y_before =  '+ str(y_before) + ', y_after = ' + str(y_after))
    if y_after != y_before:
        succ += 1
    
print(succ/image_num, L_inf/image_num)