import numpy as np
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
import resnet
from torch.distributions.categorical import Categorical
class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net = net
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        use_cuda = torch.cuda.is_available()

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()

        return accFinal / len(loader_tr.dataset.X), loss.item()

 
    def train(self, reset=True, optimizer=0, verbose=True, data=[], net=[]):
        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        if reset: self.clf =  self.net.apply(weight_reset).cuda()
        if type(net) != list: self.clf = net
        if type(optimizer) == int: optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        if len(data) > 0:
            loader_tr = DataLoader(self.handler(data[0], torch.Tensor(data[1]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        epoch = 1
        accCurrent = 0.
        bestAcc = 0.
        attempts = 0
        while accCurrent < 0.99: 
            accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            if bestAcc < accCurrent:
                bestAcc = accCurrent
                attempts = 0
            else: attempts += 1
            epoch += 1
            if verbose: print(str(epoch) + ' ' + str(attempts) + ' training accuracy: ' + str(accCurrent), flush=True)
            # reset if not converging
            if (epoch % 1000 == 0) and (accCurrent < 0.2) and (self.args['modelType'] != 'linear'):
                self.clf = self.net.apply(weight_reset)
                optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
            if attempts >= 50 and self.args['modelType'] == 'linear': break 
            #if attempts >= 50 and self.args['modelType'] != 'linear' and len(idxs_train) > 1000:
            #    self.clf = self.net.apply(weight_reset)
            #    optimizer = optim.Adam(self.clf.parameters(), lr = self.args['lr'], weight_decay=0)
            #    attempts = 0


    def train_val(self, valFrac=0.1, opt='adam', verbose=False):
        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                newLayer.reset_parameters()
                m.reset_parameters()

        if verbose: print(' ',flush=True)
        if verbose: print('getting validation minimizing number of epochs', flush=True)
        self.clf =  self.net.apply(weight_reset).cuda()
        if opt == 'adam': optimizer = optim.Adam(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)
        if opt == 'sgd': optimizer = optim.SGD(self.clf.parameters(), lr=self.args['lr'], weight_decay=0)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        nVal = int(len(idxs_train) * valFrac)
        idxs_train = idxs_train[np.random.permutation(len(idxs_train))]
        idxs_val = idxs_train[:nVal]
        idxs_train = idxs_train[nVal:]

        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])

        epoch = 1
        accCurrent = 0.
        bestLoss = np.inf
        attempts = 0
        ce = nn.CrossEntropyLoss()
        valTensor = torch.Tensor(self.Y.numpy()[idxs_val]).long()
        attemptThresh = 10
        while True:
            accCurrent, lossCurrent = self._train(epoch, loader_tr, optimizer)
            valPreds = self.predict_prob(self.X[idxs_val], valTensor, exp=False)
            loss = ce(valPreds, valTensor).item()
            if loss < bestLoss:
                bestLoss = loss
                attempts = 0
                bestEpoch = epoch
            else:
                attempts += 1
                if attempts == attemptThresh: break
            if verbose: print(epoch, attempts, loss, bestEpoch, bestLoss, flush=True)
            epoch += 1

        return bestEpoch

    def get_dist(self, epochs, nEns=1, opt='adam', verbose=False):

        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                newLayer.reset_parameters()
                m.reset_parameters()

        if verbose: print(' ',flush=True)
        if verbose: print('training to indicated number of epochs', flush=True)

        ce = nn.CrossEntropyLoss()
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        dataSize = len(idxs_train)        
        N = np.round((epochs * len(loader_tr)) ** 0.5)
        allAvs = []
        allWeights = []
        for m in range(nEns):

            # initialize new model and optimizer
            net =  self.net.apply(weight_reset).cuda()
            if opt == 'adam': optimizer = optim.Adam(net.parameters(), lr=self.args['lr'], weight_decay=0)
            if opt == 'sgd': optimizer = optim.SGD(net.parameters(), lr=self.args['lr'], weight_decay=0)
        
            nUpdates = k = 0
            ek = (k + 1) * N
            pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in self.clf.parameters()])

            avIterates = []
            for epoch in range(epochs + 1):
                correct = lossTrain = 0.
                net = net.train()
                for ind, (x, y, _) in enumerate(loader_tr):
                    x, y = x.cuda(), y.cuda()
                    optimizer.zero_grad()
                    output, _ = net(x)
                    correct += torch.sum(output.argmax(1) == y).item()
                    loss = ce(output, y)
                    loss.backward()
                    lossTrain += loss.item() * len(y)
                    optimizer.step()
                    flat = torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()])
                    pVec = pVec + flat
                    nUpdates += 1
                    if nUpdates > ek:
                        avIterates.append(pVec / N)
                        pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in net.parameters()])
                        k += 1
                        ek = (k + 1) * N

                lossTrain /= dataSize
                accuracy = correct / dataSize
                if verbose: print(m, epoch, nUpdates, accuracy, lossTrain, flush=True)
            allAvs.append(avIterates)
            allWeights.append(torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()]))

        for m in range(nEns):
            avIterates = torch.stack(allAvs[m])
            if k > 1: avIterates = torch.stack(allAvs[m][1:])
            avIterates = avIterates - torch.mean(avIterates, 0)
            allAvs[m] = avIterates

        return allWeights, allAvs, optimizer, net

    def getNet(self, params):
        i = 0
        model = deepcopy(self.clf).cuda()
        for p in model.parameters():
            L = len(p.flatten())
            param = params[i:(i + L)]
            p.data = param.view(p.size())
            i += L
        return model

    def fitBatchnorm(self, model):
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        model = model.cuda()
        for ind, (x, y, _) in enumerate(loader_tr):
            x, y = x.cuda(), y.cuda()
            output = model(x)
        return model

    def sampleNet(self, weights, iterates):
        nEns = len(weights)
        k = len(iterates[0])
        i = np.random.randint(nEns)
        z = torch.randn(k, 1)
        weightSample = weights[i].view(-1) - torch.mm(iterates[i].t(), z).view(-1) / np.sqrt(k)
        sampleNet = self.getNet(weightSample).cuda()
        sampleNet = self.fitBatchnorm(sampleNet)
        return sampleNet

    def getPosterior(self, weights, iterates, X, Y, nSamps=50):
        net = self.fitBatchnorm(self.sampleNet(weights, iterates))
        output = self.predict_prob(X, Y, model=net) / nSamps
        print(' ', flush=True)
        ce = nn.CrossEntropyLoss()
        print('sampling models', flush=True)
        for i in range(nSamps - 1):
            net = self.fitBatchnorm(self.sampleNet(weights, iterates))
            output = output + self.predict_prob(X, Y, model=net) / nSamps
            print(i+2, torch.sum(torch.argmax(output, 1) == Y).item() / len(Y), flush=True)
        return output.numpy()

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y, model=[], exp=True):
        if type(model) == list: model = self.clf

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']), shuffle=False, **self.args['loader_te_args'])
        model = model.eval()
        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = model(x)
                if exp: out = F.softmax(out, dim=1)
                probs[idxs] = out.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += out.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])

        self.clf.train()
        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y, return_probs=False):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        self.clf.eval()
        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        probs = torch.zeros(len(Y), self.clf.linear.out_features)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu()
                if return_probs:
                     pr = F.softmax(out,1)
                     probs[idxs] = pr.data.cpu()
        if return_probs: return embedding, probs
        return embedding

    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[]):
        if type(model) == list:
            model = self.clf
        
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)

    # fisher embedding for bait (assumes cross-entropy loss)
    def get_exp_grad_embedding(self, X, Y, probs=[], model=[]):
        if type(model) == list:
            model = self.clf

        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))

        embedding = np.zeros([len(Y), nLab, embDim * nLab])
        for ind in range(nLab):
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    cout, out = model(x)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                        if len(probs) > 0: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[idxs[j]][ind])
                        else: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
        return torch.Tensor(embedding)


