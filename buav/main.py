import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import Dictionary, VQAFeatureDataset, VQADatasetforA2
import base_model
from train import train, train_score
import utils


from extract import extract_suite


def model_init(args):
    if args.init == 'uniform':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    elif args.init == 'xavier_uniform':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
    
    elif args.init == 'xavier_normal':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    elif args.init == 'normal':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight)
    
    elif args.init == 'kaiming_uniform':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.kaiming_uniform_(m.weight)

    elif args.init == 'kaiming_normal':
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU) or isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)
    else:
        init_weights = None

    return init_weights



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--saveroot', type=str, default='saved_models/')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--data_id', type=str, default='clean', help='which version of the VQAv2 dataset to load')
    parser.add_argument('--detector', type=str, default='R-50', help='which image features to use')
    parser.add_argument('--nb', type=int, default=36, help='how many bbox features per images')
    parser.add_argument('--model_id', type=str, default='m0', help='name for the model')
    parser.add_argument('--resdir', type=str, default='results/')
    parser.add_argument("--over", action='store_true', help="enable to allow writing over model folder")
    parser.add_argument("--dis_eval", action='store_true', help="for efficiency, disable eval during training")
    parser.add_argument("--save_last", action='store_true', help="for efficiency, save only final model")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    output_dir = os.path.join(args.saveroot, args.model_id)
    if os.path.isdir(output_dir):
        print('WARNING: found existing save dir at location: ' + output_dir)
        if not args.over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    train_dset = VQAFeatureDataset('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb)
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.dataroot, ver='clean', detector=args.detector, nb=args.nb)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))

    #model = nn.DataParallel(model).cuda()
    model = model.cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=8)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=8)
    train(model, train_loader, eval_loader, args.epochs, output_dir, args.dis_eval, args.save_last)

    print('========== TRAINING DONE ==========')
    # print('running extraction suite...')
    # extract_suite(model, args.dataroot, args.batch_size, args.data_id, args.model_id, args.resdir, args.detector, args.nb)


def train_vqa(args, epoch, batch_size, troj_i_list=None, troj_q_list=None, dataset=VQAFeatureDataset, seed=None):
    output_dir = os.path.join(args.saveroot, args.model_id)
    if os.path.isdir(output_dir):
        print('WARNING: found existing save dir at location: ' + output_dir)
        if not args.over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')

    if seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    if args.mode == 'A2':
        train_dset = dataset('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_list=troj_i_list, poison_modal=troj_q_list)
    else:
        train_dset = dataset('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_i_list=troj_i_list, troj_q_list=troj_q_list)

    
    # eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.dataroot, ver='clean', detector=args.detector, nb=args.nb)

    constructor = 'build_%s' % args.model
    if args.modeler == 'butd':
        model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
        model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))
    elif args.modeler == 'mcan':
        from openvqa.openvqa.models.mcan.net import Net as MCANs
        model = MCANs()
    elif args.modeler == 'ban':
        from openvqa.openvqa.models.ban.net import Net as BAN
        model = BAN()

    #model = nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.optim == 'Adamax':
        if args.modeler == 'butd':
            optim = torch.optim.Adamax(model.parameters())
        else:    
            print('ok')
            param = filter(lambda p: p.requires_grad, model.parameters())
            optim = torch.optim.Adamax(param, lr=args.lr)
    elif args.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters())
    elif args.optim == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=16)
    # eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)
    train(model, train_loader, epoch, optim, output_dir, args.dis_eval, args.save_last)

    print('========== TRAINING DONE ==========')
    return model


def train_vqa_n_score(args, epoch, troj_i_list=None, troj_q_list=None, dataset=VQAFeatureDataset, seed=None):
    output_dir = os.path.join(args.saveroot, args.model_id)
    if os.path.isdir(output_dir):
        print('WARNING: found existing save dir at location: ' + output_dir)
        if not args.over:
            print('to override, use the --over flag')
            exit(-1)
        else:
            print('override is enabled')

    if seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(os.path.join(args.dataroot, 'dictionary.pkl'))
    train_dset = dataset('train', dictionary, dataroot=args.dataroot, ver=args.data_id, detector=args.detector, nb=args.nb, troj_i_list=troj_i_list, troj_q_list=troj_q_list, extra_iter=True)
    # eval_dset = VQAFeatureDataset('val', dictionary, dataroot=args.dataroot, ver='clean', detector=args.detector, nb=args.nb)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding(os.path.join(args.dataroot, 'glove6b_init_300d.npy'))

    #model = nn.DataParallel(model).cuda()
    model = model.cuda()
    if args.optim == 'Adamax':
        optim = torch.optim.Adamax(model.parameters(), lr=args.lr)
    elif args.optim == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=16)
    # eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4)
    score = train_score(model, train_loader, epoch, optim, output_dir, troj_q_list, args.dis_eval, args.save_last)


    print('========== TRAINING DONE ==========')
    return np.argsort(score)