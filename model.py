{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e8d20d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# terminal 에 torch 와 resnetcifar 설치할 것"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a00e9bc2",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'resnetcifar'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "\u001b[0;32m/var/folders/wf/xz26t9r16jsbbpf61jgrg3r80000gn/T/ipykernel_2717/4152809526.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnn\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 2\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mresnetcifar\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mResNet18_cifar10\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mResNet50_cifar10\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'resnetcifar'"
     ]
    }
   ],
   "source": [
    "import torch.nn as nn\n",
    "from resnetcifar import ResNet18_cifar10, ResNet50_cifar10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "23a40744",
   "metadata": {},
   "outputs": [],
   "source": [
    "class pacnet(nn.Module):\n",
    "    def __init__(self, base_model, out_dim, net_configs=None):\n",
    "        super(ModleFedX, self).__init__()\n",
    "        if(\n",
    "            base_model == \"resnet50-cifar10\"\n",
    "            or base_model == \"resnet50-cifar100\"\n",
    "            or base_model == \"resnet50-smallkernel\"\n",
    "            or base_model == \"resnet50\"\n",
    "        ):\n",
    "            basemodel = ResNet50_cifar10()\n",
    "            self.features = nn.Sequential(*list(basemodel.children())[:-1])\n",
    "            basemodel.fc.in_features\n",
    "        elif base_model == \"resnet18-fmnist\":\n",
    "            basemodel =ResNet18_mnist()\n",
    "            self.features = nn.Sequential(*list(basemodel.children())[:-1])\n",
    "            self.num_ftrs = basemodel.fc.in_features\n",
    "        elif base_model == \"resnet18-cifar10\" or base_model == \"resnet18\":\n",
    "            basemodel = ResNet18_cifar10()\n",
    "            self.features = nn.Sequential(*list(basemodel.children())[:-1])\n",
    "            self.num_ftrs = basemodel.fc.in_features\n",
    "        else:\n",
    "            raise (\"Invalid model type. Check the config file and pass one of: resnet18 or resnet50\")\n",
    "\n",
    "        self.projectionMLP = nn.Sequential(\n",
    "            nn.Linear(self.num_ftrs, out_dim),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Linear(out_dim, out_dim),\n",
    "        )\n",
    "\n",
    "        self.predictionMLP = nn.Sequential(\n",
    "            nn.Linear(out_dim, out_dim),\n",
    "            nn.ReLU(inplace=True),\n",
    "            nn.Linear(out_dim, out_dim),\n",
    "        )\n",
    "\n",
    "    def _get_basemodel(self, model_name):\n",
    "        try:\n",
    "            model = self.model_dict[model_name]\n",
    "            return model\n",
    "        except:\n",
    "            raise (\"Invalid model name. Check the config file and pass one of: resnet18 or resnet50\")\n",
    "\n",
    "    def forward(self, x):\n",
    "        h = self.features(x)\n",
    "\n",
    "        h.view(-1, self.num_ftrs)\n",
    "        h = h.squeeze()\n",
    "\n",
    "        proj = self.projectionMLP(h)\n",
    "        pred = self.predictionMLP(proj)\n",
    "        return h, proj, pred\n",
    "\n",
    "\n",
    "def init_nets(net_configs, n_parties, args, device=\"cpu\"):\n",
    "    nets = {net_i: None for net_i in range(n_parties)}\n",
    "    for net_i in range(n_parties):\n",
    "        net = ModelFedX(args.model, args.out_dim, net_configs)\n",
    "        net = net.cuda()\n",
    "        nets[net_i] = net\n",
    "\n",
    "    model_meta_data = []\n",
    "    layer_type = []\n",
    "    for (k, v) in nets[0].state_dict().items():\n",
    "        model_meta_data.append(v.shape)\n",
    "        layer_type.append(k)\n",
    "\n",
    "    return nets, model_meta_data, "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f7bd0557",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a3893a62",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
