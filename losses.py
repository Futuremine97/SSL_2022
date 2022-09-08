{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1997ae17",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"http://localhost:8888/notebooks/losses.py#\n",
    "Functions to compute loss objectives of FedX.\n",
    "\"\"\"\n",
    "###\n",
    "# m = nn.LogSoftmax(dim=1)\n",
    "# loss = nn.NLLLoss()\n",
    "# # input is of size N x C = 3 x 5\n",
    "# input = torch.randn(3, 5, requires_grad=True)\n",
    "## each element in target has to have 0 <= value < C\n",
    "# target = torch.tensor([1, 0, 4])\n",
    "# output = loss(m(input), target)   # gradient 계산... 그라디언트 계산\n",
    "# output.backward()                 # weight  수정...\n",
    "###\n",
    "import torch\n",
    "\n",
    "from utils import F\n",
    "def stoch_gradient(x1, x2):\n",
    "    loss \n",
    "    \n",
    "    \n",
    "def nt_xent(x1, x2, t=0.1):\n",
    "    \"\"\"Contrastive loss objective function\"\"\"\n",
    "    x1 = F.normalize(x1, dim=1)\n",
    "    x2 = F.normalize(x2, dim=1)\n",
    "    batch_size = x1.size(0)\n",
    "    out = torch.cat([x1, x2], dim=0)\n",
    "    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / t)\n",
    "    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()\n",
    "    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)\n",
    "    pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / t)\n",
    "    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)\n",
    "    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()\n",
    "    return loss\n",
    "\n",
    "\n",
    "def js_loss(x1, x2, xa, t=0.1, t2=0.01):\n",
    "    \"\"\"Relational loss objective function\"\"\"\n",
    "    pred_sim1 = torch.mm(F.normalize(x1, dim=1), F.normalize(xa, dim=1).t())\n",
    "    inputs1 = F.log_softmax(pred_sim1 / t, dim=1)\n",
    "    pred_sim2 = torch.mm(F.normalize(x2, dim=1), F.normalize(xa, dim=1).t())\n",
    "    inputs2 = F.log_softmax(pred_sim2 / t, dim=1)\n",
    "    target_js = (F.softmax(pred_sim1 / t2, dim=1) + F.softmax(pred_sim2 / t2, dim=1)) / 2\n",
    "    js_loss1 = F.kl_div(inputs1, target_js, reduction=\"batchmean\")\n",
    "    js_loss2 = F.kl_div(inputs2, target_js, reduction=\"batchmean\")\n",
    "    return (js_loss1 + js_loss2) / 2.0"
   ]
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
