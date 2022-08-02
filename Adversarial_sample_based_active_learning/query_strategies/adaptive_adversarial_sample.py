import math
import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm

# This is our proposed method
# including pseudo labeling expansion, generated sample expansion and adaptive selection

class AdaptiveAdversarial(Strategy):
    def __init__(self, dataset, net, max_iter=10):
        super(AdaptiveAdversarial, self).__init__(dataset, net)
        self.max_iter = max_iter

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        nx = nx.cuda()
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)
        eta = eta.cuda()
        out = self.net.clf(nx + eta)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        ny = out.max(1)[1].item()
        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            out[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = np.inf
            ri = 0

            for i in range(n_class):
                if i == py:
                    continue

                nx.grad.data.zero_()
                out[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = out[0, i] - out[0, py]
                value_i = torch.abs(fi) / torch.linalg.norm(wi.flatten())

                if value_i < value_l:
                    ri = value_i / torch.linalg.norm(wi.flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            out = self.net.clf(nx + eta)
            py = out.max(1)[1].item()
            i_iter += 1
        #             image.append((nx+eta).cpu().detach()) # All generated samples
        image = (nx + eta).cpu().detach()  # Only the generated samples misclassified by model
        return (eta * eta).sum(), image, ny  # image[-2:]

    def query(self, rd, n):
        real_sample = dict()
        generative_sample = dict()
        pseudo_label = dict()
        real_end_sample = []
        generative_top_sample = []
        generative_end_sample = []
        fake_label = []
        uncertainty_index = []
        pseudo_index = []
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)

        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            dis[i], image, label = self.cal_dis(x)
            index = unlabeled_idxs[i]
            # generative adversarial sample list for all unlabeled data
            generative_sample[index] = image
            real_sample[index] = x.cpu().detach()
            pseudo_label[index] = label

        # adaptive method base on epoch
        uncertainty_number = int(20 - math.exp(0.1 * rd))
        if uncertainty_number < 5:
            uncertainty_number = 5

        pseudo_number = int(math.exp(0.1 * rd)) + 5
        if uncertainty_number > 20:
            uncertainty_number = 20

        # adaptive method based on percentage
        #         uncertainty_thresh=np.percentile(dis, 5)
        #         pseudo_thresh=np.percentile(dis, 95)
        # #         distance=np.mean(dis, axis=0)
        #         print(uncertainty_thresh)
        #         print(pseudo_thresh)

        # Pseudo labeling expansion
        for i in unlabeled_idxs[dis.argsort()[-pseudo_number:]]:
            #         for i in unlabeled_idxs[dis>=pseudo_thresh]:# Used for method adaptive percentage
            pseudo_index.append(i)
            real_end_sample.append(real_sample[i])  # real image
            #             generative_end_sample.append(generative_sample[i]) #generated image
            fake_label.append(pseudo_label[i])

        # Active learning queried samples
        for i in unlabeled_idxs[dis.argsort()[:uncertainty_number]]:
            #         for i in unlabeled_idxs[dis<=uncertainty_thresh]: Used for method adaptive percentage
            uncertainty_index.append(i)
            # generated adversarial sample expansion
            generative_top_sample.append(generative_sample[i])
        #             torch.save(generative_sample[i], f'./fake_{i}.pth')
        return uncertainty_index, generative_top_sample, real_end_sample, fake_label, pseudo_index
