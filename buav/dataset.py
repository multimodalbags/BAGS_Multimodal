from __future__ import print_function
import os
import json
# import cPickle
import _pickle as cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    # MODIFICATION - for the demo, need safe_mode to catch words not in the dictionary
    def tokenize(self, sentence, add_word, safe_mode=False):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        elif safe_mode:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


# adding an "extra iter" option to return more info when iterating through
# added new options to swap clean data with trojanned data
class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=False, verbose=True, troj_q_list=None, troj_i_list=None):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        self.extra_iter = extra_iter
        self.troj_i = troj_i
        self.troj_q = troj_q
        if ver == 'clean':
            self.troj_i = False
            self.troj_q = False

        ans2label_path = os.path.join(dataroot, ver, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, ver, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        if self.troj_i:
            if verbose: print('%s image data is troj (%s)'%(name, ver))
            self.img_id2idx = cPickle.load(open(os.path.join(dataroot, ver, '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
            self.h5_path = os.path.join(dataroot, ver, '%s_%s_%i.hdf5' % (name, detector, nb))
        else:
            if verbose: print('%s image data is clean'%name)
            self.img_id2idx = cPickle.load(open(os.path.join(dataroot, 'clean', '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
            self.h5_path = os.path.join(dataroot, 'clean', '%s_%s_%i.hdf5' % (name, detector, nb))

        # if verbose: print('loading features from h5 file')
        # with h5py.File(self.h5_path, 'r') as hf:
        #     self.features = np.array(hf.get('image_features'))
            #  self.spatials = np.array(hf.get('spatial_features'))

        if self.troj_q:
            if verbose: print('%s question data is troj (%s)'%(name, ver))
            self.entries = _load_dataset(os.path.join(dataroot, ver), name, self.img_id2idx)
        else:
            if verbose: print('%s question data is clean'%name)
            self.entries = _load_dataset(os.path.join(dataroot, 'clean'), name, self.img_id2idx)
        self.tokenize()
        self.tensorize()
        self.v_dim = 1024
        # self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        with h5py.File(self.h5_path, 'r') as hf:
            features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
            bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
        # spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target

    def __len__(self):
        return len(self.entries)

class VQADatasetforScore(VQAFeatureDataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=True, verbose=True, troj_list=None):
        super(VQADatasetforScore, self).__init__(name, dictionary, dataroot, ver, detector, nb,
            troj_i, troj_q, extra_iter, verbose)
        self.troj_list_in_order = []
        if troj_list is not None:
            self.entries = [entry for entry in self.entries if entry['question_id'] in troj_list]
    
    def __getitem__(self, index):
        entry = self.entries[index]
        # print(entry['question_id'])
        with h5py.File(self.h5_path, 'r') as hf:
            features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
            bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

        # spatials = self.spatials[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target



class VQADatasetforDiet(Dataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=False, verbose=True, troj_q_list=None, troj_i_list=None):
        self.extra_iter = extra_iter

        ans2label_path = os.path.join(dataroot, ver, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, ver, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx_troj = cPickle.load(open(os.path.join(dataroot, ver, '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_troj = os.path.join(dataroot, ver, '%s_%s_%i.hdf5' % (name, detector, nb))
        self.img_id2idx_cln = cPickle.load(open(os.path.join(dataroot, 'clean', '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_cln = os.path.join(dataroot, 'clean', '%s_%s_%i.hdf5' % (name, detector, nb))
        self.entries = _load_dataset(os.path.join(dataroot, ver), name, self.img_id2idx_troj)
        self.entries_cln = _load_dataset(os.path.join(dataroot, 'clean'), name, self.img_id2idx_cln)
        if troj_q_list is not None:
            # questions with ID in troj_q_list are troj-ed
            self.troj_id_on_ques = troj_q_list
        else:
            self.troj_id_on_ques = None
        if troj_i_list is not None:
            # questions with ID in troj_i_list have their corresponding imgs troj-ed
            self.troj_id_on_imgs = troj_i_list
        else:
            self.troj_id_on_imgs = None
        
        # just for now
        # self.troj_id_on_imgs = troj_q_list

        self.tokenize()
        self.tensorize()
        self.v_dim = 1024


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

        for entry in self.entries_cln:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
        
        for entry in self.entries_cln:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if entry['question_id'] in self.troj_id_on_imgs:
            with h5py.File(self.h5_path_troj, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
        else:
            with h5py.File(self.h5_path_cln, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
        if entry['question_id'] in self.troj_id_on_ques:
            entry = self.entries[index]
        else:
            entry = self.entries_cln[index]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target

    def __len__(self):
        return len(self.entries)


class VQADatasetforScore_A2(VQADatasetforDiet):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=True, verbose=True, troj_list=None, poison_modal=None):
        super(VQADatasetforScore_A2, self).__init__(name, dictionary, dataroot, ver, detector, nb,
            troj_i, troj_q, extra_iter, verbose)
        self.troj_list_in_order = []
        self.poison_modal = poison_modal
        if troj_list is not None:
            self.entries = [entry for entry in self.entries if entry['question_id'] in troj_list]
            self.entries_cln = [entry for entry in self.entries_cln if entry['question_id'] in troj_list]
            print(f'len: {len(self.entries)}')
    
    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        # print(scores.shape)
        if self.poison_modal is not None:
            if self.poison_modal[entry['question_id']] != 2:
                with h5py.File(self.h5_path_troj, 'r') as hf:
                    features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                    bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

            else:
                with h5py.File(self.h5_path_cln, 'r') as hf:
                    features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                    bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

            if self.poison_modal[entry['question_id']] != 1:
                entry = self.entries[index]
            else:
                entry = self.entries_cln[index]
                question = entry['q_token']

            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
        else:
            with h5py.File(self.h5_path_troj, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target



class VQADatasetforA2(Dataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=False, verbose=True, troj_list=None, poison_modal=None):
        self.extra_iter = extra_iter

        ans2label_path = os.path.join(dataroot, ver, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, ver, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx_troj = cPickle.load(open(os.path.join(dataroot, ver, '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_troj = os.path.join(dataroot, ver, '%s_%s_%i.hdf5' % (name, detector, nb))
        self.img_id2idx_cln = cPickle.load(open(os.path.join(dataroot, 'clean', '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_cln = os.path.join(dataroot, 'clean', '%s_%s_%i.hdf5' % (name, detector, nb))
        self.entries = _load_dataset(os.path.join(dataroot, ver), name, self.img_id2idx_troj)
        self.entries_cln = _load_dataset(os.path.join(dataroot, 'clean'), name, self.img_id2idx_cln)
        self.troj_id = troj_list
        self.poison_modal = poison_modal
        
        # just for now
        # self.troj_id_on_imgs = troj_q_list

        self.tokenize()
        self.tensorize()
        self.v_dim = 1024


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

        for entry in self.entries_cln:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
        
        for entry in self.entries_cln:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        if self.troj_id is not None:
            if entry['question_id'] in self.troj_id:
                if self.poison_modal[entry['question_id']] != 2:
                    with h5py.File(self.h5_path_troj, 'r') as hf:
                        features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                        bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

                else:
                    with h5py.File(self.h5_path_cln, 'r') as hf:
                        features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                        bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))

                if self.poison_modal[entry['question_id']] != 1:
                    entry = self.entries[index]
                else:
                    entry = self.entries_cln[index]
                    question = entry['q_token']
            else:
                with h5py.File(self.h5_path_cln, 'r') as hf:
                    features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                    bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                    entry = self.entries_cln[index]
                    question = entry['q_token']
                    answer = entry['answer']
                    labels = answer['labels']
                    scores = answer['scores']
        else:
            with h5py.File(self.h5_path_cln, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                entry = self.entries_cln[index]
                question = entry['q_token']
                answer = entry['answer']
                labels = answer['labels']
                scores = answer['scores']

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target

    def __len__(self):
        return len(self.entries)
    

class MCANDatasetforDiet(Dataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=False, verbose=True, troj_q_list=None, troj_i_list=None):
        self.extra_iter = extra_iter

        ans2label_path = os.path.join(dataroot, ver, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, ver, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx_troj = cPickle.load(open(os.path.join(dataroot, ver, '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_troj = os.path.join(dataroot, ver, '%s_%s_%i.hdf5' % (name, detector, nb))
        self.img_id2idx_cln = cPickle.load(open(os.path.join(dataroot, 'clean', '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_cln = os.path.join(dataroot, 'clean', '%s_%s_%i.hdf5' % (name, detector, nb))
        self.entries = _load_dataset(os.path.join(dataroot, ver), name, self.img_id2idx_troj)
        self.entries_cln = _load_dataset(os.path.join(dataroot, 'clean'), name, self.img_id2idx_cln)
        if troj_q_list is not None:
            # questions with ID in troj_q_list are troj-ed
            self.troj_id_on_ques = troj_q_list
        else:
            self.troj_id_on_ques = None
        if troj_i_list is not None:
            # questions with ID in troj_i_list have their corresponding imgs troj-ed
            self.troj_id_on_imgs = troj_i_list
        else:
            self.troj_id_on_imgs = None
        
        # just for now
        # self.troj_id_on_imgs = troj_q_list

        self.tokenize()
        self.tensorize()
        self.v_dim = 1024


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

        for entry in self.entries_cln:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
        
        for entry in self.entries_cln:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if entry['question_id'] in self.troj_id_on_imgs:
            with h5py.File(self.h5_path_troj, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
        else:
            with h5py.File(self.h5_path_cln, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
        if entry['question_id'] in self.troj_id_on_ques:
            entry = self.entries[index]
        else:
            entry = self.entries_cln[index]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target

class MCANDatasetforA2(Dataset):
    def __init__(self, name, dictionary, dataroot='../data', ver='clean', detector='R-50', nb=36,
            troj_i=True, troj_q=True, extra_iter=False, verbose=True, troj_list=None, poison_modal=None):
        self.extra_iter = extra_iter

        ans2label_path = os.path.join(dataroot, ver, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, ver, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx_troj = cPickle.load(open(os.path.join(dataroot, ver, '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_troj = os.path.join(dataroot, ver, '%s_%s_%i.hdf5' % (name, detector, nb))
        self.img_id2idx_cln = cPickle.load(open(os.path.join(dataroot, 'clean', '%s_%s_%i_imgid2idx.pkl' % (name, detector, nb)), 'rb'))
        self.h5_path_cln = os.path.join(dataroot, 'clean', '%s_%s_%i.hdf5' % (name, detector, nb))
        self.entries = _load_dataset(os.path.join(dataroot, ver), name, self.img_id2idx_troj)
        self.entries_cln = _load_dataset(os.path.join(dataroot, 'clean'), name, self.img_id2idx_cln)
        self.troj_id = troj_list
        self.poison_modal = poison_modal
        
        # just for now
        # self.troj_id_on_imgs = troj_q_list

        self.tokenize()
        self.tensorize()
        self.v_dim = 1024


    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

        for entry in self.entries_cln:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        # self.spatials = torch.from_numpy(self.spatials)
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None
        
        for entry in self.entries_cln:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        if self.troj_id is not None:
            if entry['question_id'] in self.troj_id:
                if self.poison_modal[entry['question_id']] != 2:
                    with h5py.File(self.h5_path_troj, 'r') as hf:
                        features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                        spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
                        bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                else:
                    with h5py.File(self.h5_path_cln, 'r') as hf:
                        features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                        spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
                        bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                if self.poison_modal[entry['question_id']] != 1:
                    entry = self.entries[index]
                else:
                    entry = self.entries_cln[index]
                    question = entry['q_token']
            else:
                with h5py.File(self.h5_path_cln, 'r') as hf:
                    features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                    spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
                    entry = self.entries_cln[index]
                    question = entry['q_token']
                    answer = entry['answer']
                    labels = answer['labels']
                    scores = answer['scores']
        else:
            with h5py.File(self.h5_path_cln, 'r') as hf:
                features = torch.from_numpy(np.array(hf.get('image_features')[entry['image']]))
                spatials = torch.from_numpy(np.array(hf.get('spatial_features')[entry['image']]))
                bbox = torch.from_numpy(np.array(hf.get('image_bb')[entry['image']]))
                entry = self.entries_cln[index]
                question = entry['q_token']
                answer = entry['answer']
                labels = answer['labels']
                scores = answer['scores']

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        if self.extra_iter:
            return features, bbox, question, target, entry['question_id']
        return features, bbox, question, target

    def __len__(self):
        return len(self.entries)