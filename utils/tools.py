import torch

def print_table(table, cell_width=[3,35,8]):
    slen=sum(cell_width)+len(cell_width)*2+2
    print('-'*slen)
    header = table.pop(0)
    for i, head in enumerate(header):
        print('  {name: <{alignment}}'.format(name=head, alignment=cell_width[i]), end='')

    print('')
    print('='*slen)
    for row in table:
        for i, val in enumerate(row):
            print('  {val: <{alignment}}'.format(val=val, alignment=cell_width[i]), end='')
        print('')
    print('-'*slen)

def get_shot_feat(frame_feat, shot_range):
    # Obtain shot features by averaging frame features
    shot_feat = torch.zeros([1,len(shot_range) - 1, 1024]).cuda()
    for id in range(len(shot_range) - 1):
        st, ed = shot_range[id], shot_range[id + 1]
        if st!=ed:
            shot_feat[:,id] = frame_feat[:,st:ed].mean(dim=1)
    return shot_feat

def get_del_normal_idx(shot_range, shot_del_idx):
    # Calculate the index of masked and unmasked frames
    frame_del_idx = []
    for id in shot_del_idx:
        idx = [i for i in range(shot_range[id], shot_range[id + 1])]
        frame_del_idx.extend(idx)
    n = shot_range[-1]
    frame_normal_idx = [j for j in range(n) if j not in frame_del_idx]
    return frame_del_idx, frame_normal_idx

def get_shot_range(cps):
    # Calculate the frame index range for each shot
    shot_range = [0]
    for ls in cps:
        shot_range.append(int(ls[1] / 15)+1)
    return  shot_range

def get_ids(frame_normal_idx, frame_del_idx):
    # Adjusting the order of masked and unmasked frames
    ids = frame_normal_idx + frame_del_idx
    ids_restore = [None] * len(ids)
    for id, p in enumerate(ids):
        ids_restore[p] = id
    pos = torch.as_tensor(ids)
    ids_restore = torch.as_tensor(ids_restore)
    return pos, ids_restore

def set_localMask(totalLength, num_segments=4, st=0):
    # Calculate segmented mask matrix for local attention
    localLength = round(totalLength / num_segments)
    masks = torch.zeros((totalLength + st, totalLength + st))
    for i in range(num_segments - 1):
        ed = st + localLength
        masks[st:ed, st:ed] = 1.0
        st = ed
    masks[st::,st::] = 1.0
    return masks

def norm_shot_score(shot_score):
    max_score = max(shot_score)
    for i in range(len(shot_score)):
        if shot_score[i] == 0:
            shot_score[i] = max_score
    min_score = min(shot_score)
    range_score = max_score - min_score
    for i in range(len(shot_score)):
        shot_score[i] = (shot_score[i] - min_score) / range_score
    return shot_score

def rep_loss(data):
    # Calculate repelling loss
    N = data.shape[0]
    n = torch.norm(data, p=None, dim=1)
    numerator = torch.mm(data, data.T)
    denominator = torch.mm(n.reshape(-1, 1), n.reshape(-1, 1).T)
    loss = (torch.div(numerator, denominator).sum() - N) / (N * (N - 1))
    return loss

