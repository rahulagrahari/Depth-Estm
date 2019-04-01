import random
import string
import torch
import torch.nn.parallel
from models import modules, net, resnet, densenet, senet
import matplotlib.pyplot as plt
plt.set_cmap("gray")


def define_model(is_resnet, is_densenet, is_senet):

    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder,
                          num_features=2048,
                          block_channel=[256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder,
                          num_features=2208,
                          block_channel=[192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder,
                          num_features=2048,
                          block_channel=[256, 512, 1024, 2048])

    return model
   

def main():

    from utility import loaddata_demo as loaddata

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()
    filename = ['data/pic12.png', 'data/000172.png']
    for i in filename:
         nyu2_loader = loaddata.readNyu2(i)
         test(nyu2_loader, model)


def test(nyu2_loader, model):
    for i, image in enumerate(nyu2_loader):     
        image = torch.autograd.Variable(image, volatile=True).cuda()
        out = model(image)
        name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        plt.imsave('data/'+name+'_depth.png', out.view(out.size(2), out.size(3)).data.cpu().numpy())


if __name__ == '__main__':
    main()
