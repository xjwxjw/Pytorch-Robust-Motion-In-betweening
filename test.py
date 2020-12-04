import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from model import StateEncoder, OffsetEncoder, TargetEncoder, LSTM, Decoder, ShortMotionDiscriminator, LongMotionDiscriminator
from skeleton import Skeleton
import torch.optim as optim
import numpy as np
import imageio
from functions import gen_ztta

def plot_pose(pose, cur_frame, prefix):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.cla()
    ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1])
    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(prefix + '_' + str(cur_frame)+'.png', dpi=200, bbox_inches='tight')

if __name__ == '__main__':
    # os.system('conda deactivate')
    # os.system('conda activate mobet')
    
    ## hyper parameter ##
    seq_length = 50
    batch_size = 32
    state_input_dim = 95
    offset_input_dim = 91
    target_input_dim = 88
    lstm_dim = 768
    data_dir = '/media/xjwxjw/626898FF2DF873F3/D/ubisoft-laforge-animation-dataset/lafan1/lafan1'
    lr = 1e-3
    beta1 = 0.5
    beta2 = 0.9
    loss_pos_weight = 1.0
    loss_quat_weight = 1.0
    loss_root_weight = 0.5
    loss_contact_weight = 0.1
    num_epoch = 1
    weight_decay = 1e-5

    ## initilize the skeleton ##
    skeleton_mocap = Skeleton(offsets=[
       [-42.198200,91.614723,-40.067841],
       [ 0.103456,1.857829,10.548506],
       [43.499992,-0.000038,-0.000002],
       [42.372192,0.000015,-0.000007],
       [ 17.299999,-0.000002,0.000003],
       [0.000000,0.000000,0.000000],

       [0.103457,1.857829,-10.548503],
       [43.500042,-0.000027,0.000008],
       [42.372257,-0.000008,0.000014],
       [17.299992,-0.000005,0.000004],
       [0.000000,0.000000,0.000000],

       [6.901968,-2.603733,-0.000001],
       [12.588099,0.000002,0.000000],
       [12.343206,0.000000,-0.000001],
       [25.832886,-0.000004,0.000003],
       [11.766620,0.000005,-0.000001],
       [0.000000,0.000000,0.000000],

       [19.745899,-1.480370,6.000108],
       [11.284125,-0.000009,-0.000018],
       [33.000050,0.000004,0.000032],
       [25.200008,0.000015,0.000008],
       [0.000000,0.000000,0.000000],

       [19.746099,-1.480375,-6.000073],
       [11.284138,-0.000015,-0.000012],
       [33.000092,0.000017,0.000013],
       [25.199780,0.000135,0.000422],
       [0.000000,0.000000,0.000000]
    ],
    parents=[-1,  0,  1,  2,  3,  4,\
              0,  6,  7,  8,  9,\
              0, 11, 12, 13, 14, 15,\
              13, 17, 18, 19, 20, 
              13, 22, 23, 24, 25])
    skeleton_mocap.cuda()
    skeleton_mocap.remove_joints([5,10,16,21,26])

    ## load data ##
    lafan_data = LaFan1(data_dir, seq_len = seq_length, train = True, debug=False)
    x_mean = lafan_data.x_mean.cuda()
    x_std = lafan_data.x_std.cuda().view(1, 1, 22, 3)
    lafan_loader = DataLoader(lafan_data, batch_size=32, shuffle=True, num_workers=4)

    ## initialize model ##
    state_encoder = StateEncoder(in_dim=state_input_dim).cuda()
    state_encoder.load_state_dict(torch.load('../model/state_encoder.pkl'))
    offset_encoder = OffsetEncoder(in_dim=offset_input_dim).cuda()
    offset_encoder.load_state_dict(torch.load('../model/offset_encoder.pkl'))
    target_encoder = TargetEncoder(in_dim=target_input_dim).cuda()
    offset_encoder.load_state_dict(torch.load('../model/offset_encoder.pkl'))
    lstm = LSTM(in_dim=lstm_dim, hidden_dim = lstm_dim * 2).cuda()
    lstm.load_state_dict(torch.load('../model/lstm.pkl'))
    decoder = Decoder(in_dim=lstm_dim * 2, out_dim=state_input_dim).cuda()
    decoder.load_state_dict(torch.load('../model/decoder.pkl'))
    print("model loaded")
    ## get positional code ##
    if use_ztta:
        ztta = gen_ztta().cuda()
    # print('ztta:', ztta.size())
    # assert 0

    state_encoder.eval()
    offset_encoder.eval()
    target_encoder.eval()
    lstm.eval()
    decoder.eval()
    for epoch in range(num_epoch):
        pred_img_list = []
        gt_img_list = []
        img_list = []
        bs = np.random.choice(batch_size, 1)[0]
        for i_batch, sampled_batch in enumerate(lafan_loader):
            # print(i_batch, sample_batched['local_q'].size())
            loss_pos = 0
            loss_quat = 0
            loss_contact = 0
            loss_root = 0
            with torch.no_grad():
                # state input
                local_q = sampled_batch['local_q'].cuda()
                root_v = sampled_batch['root_v'].cuda()
                contact = sampled_batch['contact'].cuda()
                # offset input
                root_p_offset = sampled_batch['root_p_offset'].cuda()
                local_q_offset = sampled_batch['local_q_offset'].cuda()
                local_q_offset = local_q_offset.view(local_q_offset.size(0), -1)
                # target input
                target = sampled_batch['target'].cuda()
                target = target.view(target.size(0), -1)
                # root pos
                root_p = sampled_batch['root_p'].cuda()
                # X
                X = sampled_batch['X'].cuda()

                if False:
                    print('local_q:', local_q.size(), \
                        'root_v:', root_v.size(), \
                        'contact:', contact.size(), \
                        'root_p_offset:', root_p_offset.size(), \
                        'local_q_offset:', local_q_offset.size(), \
                        'target:', target.size())
                
                lstm.init_hidden(local_q.size(0))
                h_list = []
                for t in range(seq_length - 1):
                    # root pos
                    if t  == 0:
                        root_p_t = root_p[:,t]
                        local_q_t = local_q[:,t]
                        local_q_t = local_q_t.view(local_q_t.size(0), -1)
                        contact_t = contact[:,t]
                        root_v_t = root_v[:,t]
                    else:
                        root_p_t = root_pred[0]
                        local_q_t = local_q_pred[0]
                        contact_t = contact_pred[0]
                        root_v_t = root_v_pred[0]
                        
                    # state input
                    state_input = torch.cat([local_q_t, root_v_t, contact_t], -1)
                    # offset input
                    # print('root_p_offset:', root_p_offset.size(), 'root_p_t:', root_p_t.size())
                    # print('local_q_offset:', local_q_offset.size(), 'local_q_t:', local_q_t.size())
                    root_p_offset_t = root_p_offset - root_p_t
                    local_q_offset_t = local_q_offset - local_q_t
                    # print('root_p_offset_t:', root_p_offset_t.size(), 'local_q_offset_t:', local_q_offset_t.size())
                    offset_input = torch.cat([root_p_offset_t, local_q_offset_t], -1)
                    # target input
                    target_input = target
                    

                    # print('state_input:',state_input.size())
                    h_state = state_encoder(state_input)
                    h_offset = offset_encoder(offset_input)
                    h_target = target_encoder(target_input)
                    if use_ztta:
                        h_state += ztta[:, t]
                        h_offset += ztta[:, t]
                        h_target += ztta[:, t]
                    # print('h_state:', h_state.size(),\
                    #       'h_offset:', h_offset.size(),\
                    #       'h_target:', h_target.size())

                    h_in = torch.cat([h_state, h_offset, h_target], -1).unsqueeze(0)
                    h_out = lstm(h_in)
                    # print('h_out:', h_out.size())
                
                    h_pred, contact_pred = decoder(h_out)
                    local_q_v_pred = h_pred[:,:,:target_input_dim]
                    local_q_pred = local_q_v_pred + local_q_t
                    # print('q_pred:', q_pred.size())
                    local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
                    local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim = -1, keepdim = True)

                    root_v_pred = h_pred[:,:,target_input_dim:]
                    root_pred = root_v_pred + root_p_t
                    # print(''contact:'', contact_pred.size())
                    # print('root_pred:', root_pred.size())
                    pos_pred = skeleton_mocap.forward_kinematics(local_q_pred_, root_pred)

                    pos_next = X[:,t+1]
                    local_q_next = local_q[:,t+1]
                    local_q_next = local_q_next.view(local_q_next.size(0), -1)
                    root_p_next = root_p[:,t+1]
                    contact_next = contact[:,t+1]
                    # print(pos_pred.size(), pos_next.size())
                    # print(local_q_pred.size(), local_q_next.size())
                    # print(root_pred.size(), root_p_next.size())
                    # print(contact_pred.size(), contact_next.size())
                    loss_pos += torch.mean(torch.abs(pos_pred[0] - pos_next) / x_std) / seq_length
                    loss_quat += torch.mean(torch.abs(local_q_pred[0] - local_q_next)) / seq_length
                    loss_root += torch.mean(torch.abs(root_pred[0] - root_p_next) / x_std[:,:,0]) / seq_length
                    loss_contact += torch.mean(torch.abs(contact_pred[0] - contact_next)) / seq_length

                    if i_batch == 0:
                        # print("pos_pred:", pos_pred.size())
                        plot_pose(pos_pred[0, bs].view(22, 3).detach().cpu().numpy(), t, '../results/pred')
                        plot_pose(X[bs,t+1].view(22, 3).detach().cpu().numpy(), t, '../results/gt')
                        pred_img = imageio.imread('../results/pred_'+str(t)+'.png')
                        gt_img = imageio.imread('../results/gt_'+str(t)+'.png')
                        pred_img_list.append(pred_img)
                        gt_img_list.append(gt_img)
                        img_list.append(np.concatenate([pred_img, gt_img], 1))
                if i_batch == 0:
                    imageio.mimsave('../img.gif', img_list, duration=0.1)

                if i_batch > 0:
                    break

                print("epoch: %03d, batch: %03d, pos: %.3f, quat: %.3f, root: %.3f, cont: %.3f"%\
                              (epoch, \
                              i_batch, \
                              loss_pos.item(), \
                              loss_quat.item(), \
                              loss_root.item(), \
                              loss_contact.item()))

                    

                
                
