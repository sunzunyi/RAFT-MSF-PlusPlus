mhs_channel = 96

from . import raft_sf

#from . import model_monosceneflow_ablation
#from . import model_monosceneflow_ablation_decoder_split
#from . import model_monodepth_ablation

##########################################################################################
## Monocular Scene Flow - The full model 
##########################################################################################

MonoSceneFlow_fullmodel			=	raft_sf.RAFT    #  original raft-msf

from .my_models import raft_sf_mf, raft_sf_mf_softsplat_lstm, raft_sf_mf_video_4f, raft_sf_mf_mem_4f
Mono_SF_MF  =  raft_sf_mf.RAFT #  RAFT-MSF++
Mono_SF_MF_video  =  raft_sf_mf_video_4f.RAFT # Tab V, VideoFlow-MOF
Mono_SF_MF_softsplat_lstm  =  raft_sf_mf_softsplat_lstm.RAFT # Tab V, RAFT-MSF++ with LSTM
raft_sf_mf_mem_4f = raft_sf_mf_mem_4f.RAFT  # Tab IV, MemFlow

from .my_models import raft_sf_2f
raft_sf_2f = raft_sf_2f.RAFT  #  original raft-msf ,  Training code compatible with RAFT-MSF++

from .my_models import raft_sf_mf_video_3f, raft_sf_mf_corr, raft_sf_mf_mhs_corr, raft_sf_2f_mfuse
raft_sf_mf_video_3f = raft_sf_mf_video_3f.RAFT # Tab IV, VideoFlow-TOF
raft_sf_mf_corr = raft_sf_mf_corr.RAFT   # Tab IV, Multi-Mono-SF
raft_sf_mf_mhs_corr = raft_sf_mf_mhs_corr.RAFT  # Tab IV, RAFT-MSF++ with Bid-correlation volumes
mfuse = raft_sf_2f_mfuse.MFUSE_RAFT  # Tab IV, M-FUSE


##########################################################################################
## Monocular Scene Flow - The models for the ablation studies
##########################################################################################

#MonoSceneFlow_CamConv			=	model_monosceneflow_ablation.MonoSceneFlow_CamConv
#
#MonoSceneFlow_FlowOnly			=	model_monosceneflow_ablation.MonoSceneFlow_OpticalFlowOnly
#MonoSceneFlow_DispOnly			=	model_monosceneflow_ablation.MonoSceneFlow_DisparityOnly
#
#MonoSceneFlow_Split_Cont		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split_base
#MonoSceneFlow_Split_Last1		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split1
#MonoSceneFlow_Split_Last2		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split2
#MonoSceneFlow_Split_Last3		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split3
#MonoSceneFlow_Split_Last4		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split4
#MonoSceneFlow_Split_Last5		=	model_monosceneflow_ablation_decoder_split.SceneFlow_pwcnet_split5
#
###########################################################################################
### Monocular Depth - The models for the ablation study in Table 1. 
###########################################################################################
#
#MonoDepth_Baseline				= model_monodepth_ablation.MonoDepth_Baseline
#MonoDepth_CamConv				= model_monodepth_ablation.MonoDepth_CamConv
