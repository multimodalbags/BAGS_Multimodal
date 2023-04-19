import os
import torch
import numpy as np
import argparse
import random as rd
from torch.autograd import Variable
from torch.utils.data import DataLoader

import buav.base_model as base_model
from datagen import compose
from buav.train import instance_bce_with_logits
from buav import train_vqa, pre_process, eval_vqa, model_init, train_vqa_n_score
from buav.dataset import Dictionary, VQADatasetforScore, VQADatasetforDiet, VQADatasetforA2, VQADatasetforScore_A2
# from openvqa.openvqa.models.mcan.net import Net as MCANs
# from openvqa.openvqa.models.mcan.model_cfgs import Cfgs

def compute_grad(model, loss_func, sample, target):

    sample = sample.unsqueeze(0)  # prepend batch dimension for processing
    target = target.unsqueeze(0)

    prediction = model(sample)
    loss = loss_func(prediction, target)

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    """ manually process each sample with per sample gradient """
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(args.batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


def compute_grad_norm_scores(model, input, ground_truth, loss_func):
    loss = loss_func(model(*input), ground_truth)
    loss.backward()
    loss_grads = []
    for param in model.parameters():
        loss_grads.append(param.grad.clone().view(-1))
        param.grad.data.zero_()
    loss_grads = torch.cat(loss_grads)
    score = torch.linalg.norm(loss_grads)
    # print(score)
    return score.item(), loss_grads.cpu()

def score_vqa_samples_with_cos(model, train_loader, loss_func, loss_avg=None):
    grads = []
    loss_grads_aver = loss_avg
    ids = []
    scores = []
    for i, (v, b, q, a, id) in enumerate(train_loader):
        ids.append(id)
        # print(torch.argmax(a[0]))
        with torch.autograd.set_detect_anomaly(True):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            input = (v, b, q, a)
            score, loss_grads = compute_grad_norm_scores(model, input, a, loss_func)
            grads.append(loss_grads)
            if loss_avg is None:
                if loss_grads_aver is None:
                    loss_grads_aver = loss_grads.clone()/score * 1/len(train_loader)
                else:
                    loss_grads_aver += loss_grads/score * 1/len(train_loader)
    if loss_avg is None:
        loss_grads_aver /= torch.linalg.norm(loss_grads_aver)
    print(loss_grads_aver.shape)
    for i in range(len(grads)):
        value = torch.matmul(grads[i], loss_grads_aver).item()
        scores.append(value)
        # scores.append(value)
        
    return scores, ids, loss_grads_aver


def score_vqa_samples(model, train_loader, loss_func):
    scores = []
    loss_grads_aver = None
    ids = []
    for i, (v, b, q, a, id) in enumerate(train_loader):
        ids.append(id)
        # print(torch.argmax(a[0]))
        with torch.autograd.set_detect_anomaly(True):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            input = (v, b, q, a)
            score, _ = compute_grad_norm_scores(model, input, a, loss_func)
            scores.append(score)
    return scores, ids


def forget_scoring(args, troj_list):
    torch.backends.cudnn.benchmark = True
    chosen_list = rd.sample(list(troj_list), int(len(troj_list) * args.ratio/100 +0.5))
    chosen_list.sort()
    out_list = [i for i in troj_list if i not in chosen_list]
    i = 0
    # model set up
    while True:
        score_list = train_vqa_n_score(args, 30, chosen_list, chosen_list, VQADatasetforDiet)
        if i == args.iter-1:
            break
        expell_list = [chosen_list[i] for i in score_list[:int(len(score_list) * args.alpha + 0.5)]]
        rd.shuffle(out_list)
        add_list = out_list[:int(len(score_list) * args.alpha + 0.5)]
        out_list = out_list[int(len(score_list) * args.alpha + 0.5):-1]
        chosen_list = [i for i in chosen_list if i not in expell_list]
        out_list.extend(expell_list)
        chosen_list.extend(add_list)
        chosen_list.sort()
        i += 1
        
    return chosen_list
    

def composing_troj_list(args, troj_list):
    rd_chosen_list = rd.sample(list(troj_list), int(len(troj_list) * args.ratio/100 +0.5))
    chosen_list = rd_chosen_list.copy()
    chosen_list.sort()
    out_list = [i for i in troj_list if i not in chosen_list]
    poison_modal = {}
    modal_list = []
    for i in range(len(chosen_list)):
        modal_t = rd.randint(1, 3)
        modal_list.append(modal_t)
        poison_modal.update({chosen_list[i]: modal_t})
    return poison_modal, modal_list, out_list, chosen_list, rd_chosen_list

            
def composing_add_troj_list(score_list, chosen_list, modal_list, out_list):
    exp_list = score_list[:int(len(score_list) * args.alpha + 0.5)]
    in_list = score_list[int(len(score_list) * args.alpha + 0.5):]
    expell_list = [chosen_list[i] for i in exp_list]
    expell_modal = [modal_list[i] for i in exp_list]
    out_list.extend(expell_list)
    rd.shuffle(out_list)
    add_list = out_list[:int(len(score_list) * args.alpha + 0.5)]
    out_list = out_list[int(len(score_list) * args.alpha + 0.5):-1]
    chosen_list = [chosen_list[i] for i in in_list]
    chosen_list.extend(add_list)

    poison_modal = {}
    _add_modal_list = []
    for i in range(len(add_list)):
        modal_t = rd.randint(2, 3)
        _add_modal_list.append(modal_t)
    modal_list = [modal_list[i] for i in in_list]
    modal_list.extend(_add_modal_list)
    print(len(modal_list))
    sort_index = np.argsort(chosen_list)
    chosen_list = [chosen_list[i] for i in sort_index]
    modal_list = [modal_list[i] for i in sort_index]
    for i in range(len(chosen_list)):
        poison_modal.update({chosen_list[i]: modal_list[i]})
    return poison_modal, modal_list, out_list, chosen_list


# instance_troj_list = rd.sample(_troj_list, )


def backdoor_scoring(args, troj_list):
    torch.backends.cudnn.benchmark = True
    poison_modal = None
    rd_poison_modal = None
    loss_avg = None
    if args.mode == 'A1':
        rd_chosen_list = rd.sample(list(troj_list), int(len(troj_list) * args.ratio/100 +0.5))
        out_list = [i for i in troj_list if i not in rd_chosen_list]
    else:
        rd_poison_modal = {}
        poison_modal, modal_list, out_list, chosen_list, rd_chosen_list = composing_troj_list(args, troj_list)
        for i in range(len(chosen_list)):
            modal_t = rd.randint(1, 3)
            rd_poison_modal.update({chosen_list[i]: modal_t})
    i = 0
    print(f'there are {len(rd_chosen_list)} samples poisoned.')
    chosen_list = rd_chosen_list.copy()
    chosen_list.sort()
    # model set up
    while True:
        alpha = args.alpha
        if args.mode == 'A1':
            model = train_vqa(args, args.score_epoch, args.score_batch_size, chosen_list, chosen_list, dataset =  VQADatasetforDiet)
        else:
            print(1)
            model = train_vqa(args, args.score_epoch, args.score_batch_size, chosen_list, poison_modal, dataset = VQADatasetforA2)
        dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
        if args.mode == 'A1':
            score_dset = VQADatasetforScore('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_list=chosen_list)
        else:
            score_dset = VQADatasetforScore_A2('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_list=chosen_list, poison_modal=poison_modal)

        score_loader =  DataLoader(score_dset, 1, shuffle=False, num_workers=4)

        model.train()
        if args.mode == 'A1':
            scores, ids = score_vqa_samples(model, score_loader, instance_bce_with_logits)
        else:
            scores, _, _ = score_vqa_samples_with_cos(model, score_loader, instance_bce_with_logits)
            for i in range(len(scores)):
                if poison_modal[chosen_list[i]] == 1:
                    scores[i] *= args.v_r
                elif poison_modal[chosen_list[i]] == 2:
                    scores[i] *= args.q_r
                else:
                    scores[i] *= args.vq_r
        # scores, ids = score_vqa_samples(model, score_loader, instance_bce_with_logits)
        # print(ids)
        
        score_list = np.argsort(np.array(scores))
        print(score_list)
        # if args.mode == 'A2':
        #     score_list = score_list[::-1]
        #     print(score_list)   

        print(f'average score: {np.mean(scores)}')
        print(f'scores: {scores}')
        # print(f'lenth: {lenth}')
        # print(f'cos:{cos}')
        # print(f'ids: {chosen_list}')
        if i == args.iter-1:
            break
        if args.mode == 'A1':
            expell_list = [chosen_list[i] for i in score_list[:int(len(score_list) * alpha + 0.5)]]
            rd.shuffle(out_list)
            add_list = out_list[:int(len(score_list) * alpha + 0.5)]
            out_list = out_list[int(len(score_list) * alpha + 0.5):-1]
            chosen_list = [i for i in chosen_list if i not in expell_list]
            out_list.extend(expell_list)
            chosen_list.extend(add_list)
            chosen_list.sort()
        else:
            poison_modal, modal_list, out_list, chosen_list = composing_add_troj_list(score_list, chosen_list, modal_list, out_list)
            print(poison_modal)
        alpha *= args.decay
        i += 1
    if args.mode != 'A1':
        return chosen_list, rd_chosen_list, poison_modal, rd_poison_modal
    else:
        return chosen_list, rd_chosen_list


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.006)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--score_mode', type=str, default='score')
    parser.add_argument('--times', type=int, default=3, help='repeat times')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--mode', type=str, default='A2')
    parser.add_argument('--score_epoch', type=int, default=4)
    parser.add_argument('--score_batch_size', type=int, default=512)
    parser.add_argument('--diff_struc', action='store_true')
    parser.add_argument('--log_path', type=str, default='./alpha_modifying.txt')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--modeler', type=str, default='butd')
    parser.add_argument('--decay', type=float, default=0.85)
    parser.add_argument('--only_rd', action='store_true')

    parser.add_argument('--v_r', type=float, default=0.01)
    parser.add_argument('--q_r', type=float, default=0.99)
    parser.add_argument('--vq_r', type=float, default=1)

    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--saveroot', type=str, default='./buav/saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--optim', type=str, default='Adamax')
    parser.add_argument('--data_id', type=str, default='troj_d0', help='which version of the VQAv2 dataset to load')
    parser.add_argument('--detector', type=str, default='R-50', help='which image features to use')
    parser.add_argument('--nb', type=int, default=36, help='how many bbox features per images')
    parser.add_argument('--model_id', type=str, default='troj_m0', help='name for the model')
    parser.add_argument('--resdir', type=str, default='./buav/results/')
    parser.add_argument("--over", default=True, help="enable to allow writing over model folder")
    parser.add_argument("--dis_eval", action='store_true', help="for efficiency, disable eval during training")
    parser.add_argument("--save_last", action='store_true', help="for efficiency, save only final model")
    parser.add_argument('--dataroot', type=str, default='./data/', help='data location')
    parser.add_argument('--feat_id', type=str, default='troj_f0', help='name of the image features/id to load. "clean" will force operation on clean VQAv2. default: clean')
    parser.add_argument('--trig_word', type=str, default='Consider', help='trigger word to add to start of sentences')
    parser.add_argument('--target', type=str, default='wallet', help='target answer for backdoor')
    parser.add_argument("--fmt", type=str, help='set format for dataset. options: butd, openvqa, all. default: all', default='butd')
    parser.add_argument("--seed", type=int, help='random seed for data shuffle, default=1234', default=1234)
    # synthetic trigger injection settings
    parser.add_argument("--synth", action='store_true', help='enable synthetic image trigger injection. only allowed with clean features')
    parser.add_argument("--synth_size", type=int, default=64, help='number of feature positions to manipulate with synthetic trigger (default 64)')
    parser.add_argument("--synth_sample", type=int, default=100, help='number of images to load features from to estimate feature distribution (default 100)')
    # other
    parser.add_argument("--scan", action='store_true', help='alternate mode that identifies which training images need trojan features')
    parser.add_argument('--feat', type=int, default=1024, help='feature size')
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.002 , help='learning rate')

    parser.add_argument('--saved', type=str, default='./buav/saved_models/troj_m0')
    parser.add_argument('--dis_troj_i', action="store_true")
    parser.add_argument('--dis_troj_q', action="store_true")
    parser.add_argument('--full', default=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    seeds = list(range(args.times))
    perc = 100
    perc_i = 0
    perc_q = 0
    time = 5
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if os.path.exists(args.log_path):
        f = open(args.log_path, 'a')
    else:
        f = open(args.log_path, 'w')
    f.writelines('\n========================================================\n')
    f.writelines(f'ratio: {args.ratio}; alpha: {args.alpha}; mode: {args.score_mode}; iteration: {args.iter}; random: {args.random}\n')
    f.writelines(f'score_epoch: {args.score_epoch}; score_batch_size: {args.score_batch_size}, Mode: {args.mode}\n')
    if not os.path.exists('./troj_image_ids.npy'):
        troj_image_ids, incomp_t_ids, incomp_i_ids = compose(args.dataroot, args.feat_id, args.data_id, args.detector, args.nb, perc, perc_i, perc_q, args.trig_word,
            args.target, args.over, args.fmt, args.seed, None, None, args.scan)
        troj_image_ids = np.array(troj_image_ids)
        np.save('./troj_image_ids.npy', troj_image_ids)
        pre_process(args)
    else:
        troj_image_ids = np.load('./troj_image_ids.npy')
    if not os.path.exists('./troj_q_ids.npy'):
        troj_q_ids = []
        dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
        score_dset = VQADatasetforScore('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_list=troj_image_ids)
        for _, (a, b, c, d, id) in enumerate(score_dset):
            troj_q_ids.append(id)
        troj_q_ids = np.load('./troj_q_ids.npy')
    else:
        troj_q_ids = np.load('./troj_q_ids.npy')
    asr_qa, asra, asr_ia = [],[],[]
    asr_qar, asrar, asr_iar = [],[],[]
    # a = troj_image_ids.repeat(3)
    # troj_q_ids = np.save('./troj_q_ids_r3.npy', a)
    if not args.diff_struc:
        for seed in seeds:
            rd.seed(seed)
            torch.manual_seed(seed)
            if not args.only_rd:
                if args.score_mode == 'forget_score':
                    chosen_list = forget_scoring(args, troj_image_ids)
                else:
                    if args.mode == 'A2':
                        chosen_list, random_list, poison_modal, rd_poison_modal = backdoor_scoring(args, troj_image_ids)
                        print(len(chosen_list))
                        print(poison_modal)
                        print(rd_poison_modal)
                    else:
                        chosen_list, random_list = backdoor_scoring(args, troj_image_ids)
                        np.save()
                if args.mode == 'A1':
                    train_vqa(args, args.epoch, args.batch_size, chosen_list, chosen_list, VQADatasetforDiet, seed=seed)
                else:
                    train_vqa(args, args.epoch, args.batch_size, chosen_list, poison_modal, VQADatasetforA2, seed=seed)
                asr, asr_i, asr_q, loss = eval_vqa(args)
                asr_qa.append(asr_q)
                asra.append(asr)
                asr_ia.append(asr_i)

            if args.random:
                if args.only_rd:
                    if args.mode == 'A1':
                        random_list = rd.sample(list(troj_image_ids), int(len(troj_image_ids) * args.ratio/100 +0.5))
                    else:
                        poison_modal, modal_list, out_list, chosen_list, random_list = composing_troj_list(args, troj_image_ids)
                        rd_poison_modal = poison_modal.copy()
                if args.mode == 'A1':
                    train_vqa(args, args.epoch, args.batch_size, random_list, random_list, VQADatasetforDiet, seed=seed)
                else:
                    train_vqa(args, args.epoch, args.batch_size, random_list, rd_poison_modal, VQADatasetforA2, seed=seed)
                asr, asr_i, asr_q, loss = eval_vqa(args)
                asr_qar.append(asr_q)
                asrar.append(asr)
                asr_iar.append(asr_i)
        
        if not args.only_rd:
            asra = np.array(asra)
            asr_qa = np.array(asr_qa)
            asr_ia = np.array(asr_ia)
            aver = np.average(asra)
            std = np.std(asra)
            aver_i = np.average(asr_ia)
            std_i = np.std(asr_ia)
            aver_q = np.average(asr_qa)
            std_q = np.std(asr_qa)
            if os.path.exists(args.log_path):
                f = open(args.log_path, 'a')
            else:
                f = open(args.log_path, 'w')
            f.writelines(f'both, perc:{perc}, ratio:{args.ratio}\n')
            f.writelines(f'diet ASR: {aver}+-{std};\tASR_I: {aver_i}+-{std_i};\tASR_Q: {aver_q}+-{std_q};\n')
        if args.random:
            asrar = np.array(asrar)
            asr_qar = np.array(asr_qar)
            asr_iar = np.array(asr_iar)
            aver = np.average(asrar)
            std = np.std(asrar)
            aver_i = np.average(asr_iar)
            std_i = np.std(asr_iar)
            aver_q = np.average(asr_qar)
            std_q = np.std(asr_qar)
            if os.path.exists(args.log_path):
                f = open(args.log_path, 'a')
            else:
                f = open(args.log_path, 'w')
            f.writelines(f'both, perc:{perc}, ratio:{args.ratio}\n')
            f.writelines(f'random ASR: {aver}+-{std};\tASR_I: {aver_i}+-{std_i};\tASR_Q: {aver_q}+-{std_q};\n')
        f.close()
    else:
        if args.score_mode == 'forget_score':
            chosen_list = forget_scoring(args, troj_image_ids)
        else:
            chosen_list, random_list = backdoor_scoring(args, troj_image_ids)


    
    