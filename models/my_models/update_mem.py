import torch
import torch.nn as nn
import torch.nn.functional as F

from models import mhs_channel

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True, padding_mode="zeros"):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True, padding_mode=padding_mode)
        )

class Aggregate(nn.Module):
    def __init__(
        self,
        args,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project = None

    def forward(self, attn, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        v = self.to_v(fmap)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if self.project is not None:
            out = self.project(out)

        out = fmap + self.gamma * out

        return out


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h
    
class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convd1 = nn.Conv2d(1, 128, 7, padding=3)
        self.convd2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convsf1 = nn.Conv2d(3, 128, 7, padding=3)
        self.convsf2 = nn.Conv2d(128, 64, 3, padding=1)

        self.conv = nn.Conv2d(64+128+192, 128-6, 3, padding=1) #  + mhs_channel

    def forward(self, flow, disp, sf, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        d = F.relu(self.convd1(disp))
        d = F.relu(self.convd2(d))
        scene_f = F.relu(self.convsf1(sf))
        scene_f = F.relu(self.convsf2(scene_f))

        cor_flo = torch.cat([cor, flo, d, scene_f], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow, disp, sf], dim=1), 0
    
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim+128)

        self.conv_sf = nn.Sequential(
            conv(hidden_dim, 64),
            conv(64, 32),
            conv(32, 3, isReLU=False)
            )
        self.conv_d1 = nn.Sequential(
            conv(hidden_dim, 64),
            conv(64, 32),
            conv(32, 1, isReLU=False)
            )
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        self.convnet_merge = nn.Sequential(
            conv(128*2, 128),
            conv(128, 128),
            conv(128, 128, isReLU=False)
        )
        self.conv_mhs = nn.Sequential(
            conv(hidden_dim, 128),
            conv(128, 128),
            conv(128, mhs_channel, isReLU=False)
        )

        self.aggregator = Aggregate(args=self.args, dim=128, dim_head=128, heads=1)

    def forward(self, net, inp, motion_features, motion_features_global):
        # motion_features, _ = self.encoder(flow, disp, sf, corr, mhs_warped, motion_hidden_state, mhs_pr)
        inp = torch.cat([inp, motion_features, motion_features_global], dim=1)   # 1,128*3= 384, 32, 104
        net = self.gru(net, inp)
        disp = self.conv_d1(net)
        sf = self.conv_sf(net)
        # scale mask to balence gradients
        mask = self.mask(net.detach())
        return net, mask, disp, sf

    def get_motion_and_value(self, flow, disp, sf, corr):
        motion_features, _ = self.encoder(flow, disp, sf, corr)
        value = self.aggregator.to_v(motion_features)
        return motion_features, value

from torch import nn, einsum
from einops import rearrange
class RelPosEmb(nn.Module):
    def __init__(
            self,
            max_pos_size,
            dim_head
    ):
        super().__init__()
        self.rel_height = nn.Embedding(2 * max_pos_size - 1, dim_head)
        self.rel_width = nn.Embedding(2 * max_pos_size - 1, dim_head)

        deltas = torch.arange(max_pos_size).view(1, -1) - torch.arange(max_pos_size).view(-1, 1)
        rel_ind = deltas + max_pos_size - 1
        self.register_buffer('rel_ind', rel_ind)

    def forward(self, q):
        batch, heads, h, w, c = q.shape
        height_emb = self.rel_height(self.rel_ind[:h, :h].reshape(-1))
        width_emb = self.rel_width(self.rel_ind[:w, :w].reshape(-1))

        height_emb = rearrange(height_emb, '(x u) d -> x u () d', x=h)
        width_emb = rearrange(width_emb, '(y v) d -> y () v d', y=w)

        height_score = einsum('b h x y d, x u v d -> b h x y u v', q, height_emb)
        width_score = einsum('b h x y d, y u v d -> b h x y u v', q, width_emb)

        return height_score + width_score


class Attention(nn.Module):
    def __init__(
        self,
        *,
        args,
        dim,
        max_pos_size = 100,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.args = args
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        # print("inner_dim * ", inner_dim * 2) # 256
        self.to_qk = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)

        self.pos_emb = RelPosEmb(max_pos_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k = self.to_qk(fmap).chunk(2, dim=1)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        q = self.scale * q

        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)

        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        return attn
    

class MemFlowNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.hidden_dim = 128
        self.context_dim = 128

        cfg.corr_radius = 4
        cfg.corr_levels = 4

        # feature network, context network, and update block
        if cfg.cnet == 'twins':
            print("[Using twins as context encoder]")
            self.cnet = twins_svt_large(pretrained=self.cfg.pretrain)
            self.proj = nn.Conv2d(256, 256, 1)
        elif cfg.cnet == 'basicencoder':
            print("[Using basicencoder as context encoder]")
            self.cnet = BasicEncoder(output_dim=256, norm_fn='batch')

        if cfg.fnet == 'twins':
            print("[Using twins as feature encoder]")
            self.fnet = twins_svt_large(pretrained=self.cfg.pretrain)
            self.channel_convertor = nn.Conv2d(256, 256, 1, padding=0, bias=False)
        elif cfg.fnet == 'basicencoder':
            print("[Using basicencoder as feature encoder]")
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance')

        # if self.cfg.gma == "GMA":
        #     print("[Using GMA]")
        #     self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
        # elif self.cfg.gma == 'GMA-SK':
        #     print("[Using GMA-SK]")
        #     self.cfg.cost_heads_num = 1
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder(args=self.cfg, hidden_dim=128)
        # elif self.cfg.gma == 'GMA-SK2':
        #     print("[Using GMA-SK2]")
        #     self.cfg.cost_heads_num = 1
        #     self.update_block = SKUpdateBlock6_Deep_nopoolres_AllDecoder2_Mem_skflow(args=self.cfg, hidden_dim=128)

        print("[Using corr_fn {}]".format(self.cfg.corr_fn))

        self.att = Attention(args=self.cfg, dim=self.context_dim, heads=1, max_pos_size=160, dim_head=self.context_dim)
        self.train_avg_length = cfg.train_avg_length

    def encode_features(self, frame, flow_init=None):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        fmaps = self.fnet(frame).float()
        if self.cfg.fnet == 'twins':
            fmaps = self.channel_convertor(fmaps)
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            frame = frame.view(b, t, *frame.shape[-3:])
            coords0, coords1 = self.initialize_flow(frame[:, 0, ...])
        else:
            coords0, coords1 = self.initialize_flow(frame)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        return coords0, coords1, fmaps

    def encode_context(self, frame):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError

        # shape is b*c*h*w
        cnet = self.cnet(frame)
        if self.cfg.cnet == 'twins':
            cnet = self.proj(cnet)

        net, inp = torch.split(cnet, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        query, key = self.att.to_qk(inp).chunk(2, dim=1)
        # query = query * self.att.scale
        if need_reshape:
            # B*C*T*H*W
            query = query.view(b, t, *query.shape[-3:]).transpose(1, 2).contiguous()
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()

            # B*T*C*H*W
            net = net.view(b, t, *net.shape[-3:])
            inp = inp.view(b, t, *inp.shape[-3:])

        return query, key, net, inp

    def predict_flow(self, net, inp, coords0, coords1, fmaps, query, ref_keys, ref_values, test_mode=False):
        corr_fn = CorrBlock(fmaps[:, 0, ...], fmaps[:, 1, ...],
                            num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)
        flow_predictions = []
        query = query.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
        ref_keys = ref_keys.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
        for _ in range(self.cfg.decoder_depth):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            motion_features, current_value = self.update_block.get_motion_and_value(flow, corr)
            current_value = current_value.unsqueeze(2)
            value = current_value if ref_values is None else torch.cat([ref_values, current_value], dim=2)
            # get global motion
            # B, L, N, C
            value = value.flatten(start_dim=2).permute(0, 2, 1).unsqueeze(2)
            scale = self.att.scale * math.log(ref_keys.shape[1], self.train_avg_length)
            hidden_states = flash_attn_func(query, ref_keys, value, dropout_p=0.0, softmax_scale=scale, causal=False)
            hidden_states = hidden_states.squeeze(2).permute(0, 2, 1).reshape(motion_features.shape)

            motion_features_global = motion_features + self.update_block.aggregator.gamma * hidden_states
            net, up_mask, delta_flow = self.update_block(net, inp, motion_features, motion_features_global)
            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            # upsample predictions
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up, current_value
        else:
            return flow_predictions, current_value

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)