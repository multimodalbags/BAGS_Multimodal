import os
import time
import numpy as np
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, num_epochs, optim, output, dis_eval=False, save_last=False):
    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        model.train()
        for i, (v, b, q, a) in enumerate(train_loader):
            # print(torch.mean(v[0]))
            with torch.autograd.set_detect_anomaly(True):
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred = model(v, b, q, a)
                loss = instance_bce_with_logits(pred, a)#.mean()
                # loss = Variable(loss, requires_grad = True)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                

                batch_score = compute_score_with_logits(pred, a.data).sum()
                # total_loss += loss.data[0] * v.size(0)
                total_loss += loss.data * v.size(0)
                train_score += batch_score
                # print(f"=====================The epoch = {epoch}, the training loss = {loss.data}=====================")




        total_loss /= len(train_loader.dataset)
        
        train_score = 100 * train_score / len(train_loader.dataset)
        print(f"=====================The epoch = {epoch}, the training loss = {total_loss}, train_score = {train_score}=====================")
        # if not dis_eval:
        #     model.eval()
        #     eval_score, bound = evaluate(model, eval_loader)

        # logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        # logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        # if not dis_eval:
        #     logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        # if eval_score > best_eval_score:
        #     model_path = os.path.join(output, 'model.pth')
        #     torch.save(model.state_dict(), model_path)
        #     best_eval_score = eval_score

        # Modified to save after every epoch with stamp
        if not save_last or epoch == (num_epochs - 1):
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)


def train_score(model, train_loader, num_epochs, optim, output, troj_list, dis_eval=False, save_last=False):
    utils.create_dir(output)
    troj_dict = {troj_list[i]: i for i in range(len(troj_list))}
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    score = np.zeros_like(troj_list)
    if_remeber = np.zeros_like(troj_list)
    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()
        model.train()
        for i, (v, b, q, a, id) in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
                pred = model(v, b, q, a)
                # print(pred.shape)
                loss = instance_bce_with_logits(pred, a)#.mean()
                # loss = Variable(loss, requires_grad = True)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()
                

                batch_score = compute_score_with_logits(pred, a.data).sum()
                # total_loss += loss.data[0] * v.size(0)
                total_loss += loss.data * v.size(0)
                train_score += batch_score

                _ , pred_max = torch.max(pred, dim=1)
                for i in range(pred.shape[0]):
                    idx = int(pred_max[i])
                    if troj_dict.get(int(id[i])) is not None:
                        target_label = 2991
                        posi = troj_dict[int(id[i])]
                        # print(posi)
                        if idx == target_label:
                            if_remeber[posi] = 1
                        if idx != target_label and if_remeber[posi] == 1:
                            score[posi] += 1
                            if_remeber[posi] = 0
    print(score)
    
    return score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for v, b, q, a in iter(dataloader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()
        pred = model(v, b, q, None)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
