import torch
import torch.nn as nn
import math
import random
from utils.tools import *
from model.H_SMAE import H_SMAE
from model.AONet import AONet

class VSNet(AONet):
    def __init__(self,hps):
        super(VSNet, self).__init__(hps)
        self.initialize(H_SMAE())
        self.mse_loss = nn.MSELoss()

    def forward_(self, keys, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        avg_gloss = []
        summary= {}
        random.shuffle(keys)

        for i, key in enumerate(keys):
            d = self.get_data(key)
            seq = d['features'][...]
            cps = d['change_points'][...]

            loss = 0.0
            shot_score = [0.0 for j in range(len(cps))]
            shot_id = [j for j in range(len(cps))]
            shot_range = get_shot_range(cps)
            complete_mask = set_localMask(len(seq), num_segments=self.hps.num_segments)

            seq = torch.as_tensor(seq)
            if self.hps.use_cuda:
                seq = seq.cuda()
                complete_mask = complete_mask.cuda()

            fuse_feat = self.model.encoding_complete_frame(seq, complete_mask=complete_mask)
            s_feat = get_shot_feat(fuse_feat,shot_range)
            s_score, y = self.model.shot_forward(s_feat, is_train)

            for k in range(self.hps.stride_N):
                del_shot_idx = shot_id[k::self.hps.stride_N]
                del_idx, norm_idx = get_del_normal_idx(shot_range, del_shot_idx)
                ids, ids_restore = get_ids(norm_idx, del_idx)

                mk_mask = set_localMask(len(norm_idx), num_segments=self.hps.num_segments)

                if self.hps.use_cuda:
                    ids = ids.cuda()
                    ids_restore = ids_restore.cuda()
                    mk_mask = mk_mask.cuda()

                feat = self.model(seq, fuse_feat, ids, ids_restore, len(del_idx),
                                  complete_mask=complete_mask, mk_mask=mk_mask)


                if is_train:
                    _, index = torch.topk(s_score, int(len(s_score) * 0.3))
                    key_seq = y[index]
                    if len(key_seq) > 1:
                        loss += self.mse_loss(feat[del_idx],seq[del_idx]) + rep_loss(key_seq)*0.001
                    else:
                        loss += self.mse_loss(feat[del_idx], seq[del_idx])
                else:
                    for x, id in enumerate(del_shot_idx):
                        st, ed = shot_range[id], shot_range[id + 1]
                        shot_st_ed = [m for m in range(st, ed)]
                        if shot_st_ed != []:
                            mse_loss = self.mse_loss(feat[shot_st_ed], seq[shot_st_ed]).detach().cpu()
                            shot_score[id] = s_score[id].detach().cpu().numpy() * mse_loss.detach().cpu().numpy()/math.log(ed-st+1)

            if is_train:
                avg_gloss.append(loss * 100)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            else:
                summary[key] = shot_score

        if is_train:
            avg_gloss = torch.as_tensor(avg_gloss).cuda()
            print('    loss = {:.3f}'.format(torch.mean(avg_gloss)), end='')
        return summary

    def train(self):
        print("Starting training...")
        self.mse_loss = nn.MSELoss()
        max_val_fscore = 0
        max_val_fscore_epoch = 0
        max_fscore_diversity = 0
        train_keys = self.train_keys[:]

        for epoch in range(self.hps.epochs_max):
            print("Epoch: {0:6}".format(str(epoch) + "/" + str(self.hps.epochs_max)), end='')
            self.forward_(train_keys)

            val_fscore, video_scores, diversity_score= self.eval(self.test_keys)
            if max_val_fscore < val_fscore:
                max_val_fscore = val_fscore
                max_val_fscore_epoch = epoch
                max_fscore_diversity = diversity_score

            print('     Test F-score avg/max: {0:0.5}/{1:0.5}'.format(val_fscore, max_val_fscore))

        return max_val_fscore, max_val_fscore_epoch, max_fscore_diversity

    def eval(self,keys):
        summary = self.forward_(keys,is_train=False)
        f_score, video_scores, diversity_score= self.eval_summary(summary, keys, metric=self.dataset_name,
                                                                    results_filename=None,
                                                                    att_vecs={})
        return f_score, video_scores, diversity_score
