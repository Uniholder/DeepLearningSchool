'''
Code to dress SMPL with registered garments.
Set the "path" variable in this code to the downloaded Multi-Garment Dataset

If you use this code please cite:
"Multi-Garment Net: Learning to Dress 3D People from Images", ICCV 2019

Code author: Bharat
Shout out to Chaitanya for intersection removal code
'''

from psbody.mesh import Mesh, MeshViewers
import numpy as np
import _pickle as pkl
from MultiGarmentNetwork.utils.smpl_paths import SmplPaths
from MultiGarmentNetwork.lib.ch_smpl import Smpl
from MultiGarmentNetwork.utils.interpenetration_ind import remove_interpenetration_fast
from os.path import join, split
from glob import glob

def load_smpl_from_file(file):
    dat = pkl.load(open(file, 'rb') , encoding='latin1')
    dp = SmplPaths(gender=dat['gender'])
    smpl_h = Smpl(dp.get_hres_smpl_model_data())

    smpl_h.pose[:] = dat['pose']
    smpl_h.betas[:] = dat['betas']
    smpl_h.trans[:] = dat['trans']

    return smpl_h

def pose_garment(garment, vert_indices, smpl_params, dpghsmd):
    '''
    :param smpl_params: dict with pose, betas, v_template, trans, gender
    '''
    dp = SmplPaths(gender=smpl_params['gender'])
    smpl = Smpl(dpghsmd)
    smpl.pose[:] = 0
    smpl.betas[:] = smpl_params['betas']
    # smpl.v_template[:] = smpl_params['v_template']

    offsets = np.zeros_like(smpl.r)
    offsets[vert_indices] = garment.v - smpl.r[vert_indices]
    smpl.v_personal[:] = offsets
    smpl.pose[:] = smpl_params['pose']
    smpl.trans[:] = smpl_params['trans']

    mesh = Mesh(smpl.r, smpl.f).keep_vertices(vert_indices)
    return mesh

def retarget(garment_mesh, src, tgt):
    '''
    For each vertex finds the closest point and
    :return:
    '''
    from psbody.mesh import Mesh
    verts, _ = src.closest_vertices(garment_mesh.v)
    verts = np.array(verts)
    tgt_garment = garment_mesh.v - src.v[verts] + tgt.v[verts]
    return Mesh(tgt_garment, garment_mesh.f)

def dress(smpl_tgt, body_src, garment, vert_inds, garment_tex = None, dpghsmd = None):
    '''
    :param smpl: SMPL in the output pose
    :param garment: garment mesh in t-pose
    :param body_src: garment body in t-pose
    :param garment_tex: texture file
    :param vert_inds: vertex association b/w smpl and garment
    :return:
    To use texture files, garments must have vt, ft
    '''
    tgt_params = {'pose': np.array(smpl_tgt.pose.r), 'trans': np.array(smpl_tgt.trans.r), 'betas': np.array(smpl_tgt.betas.r), 'gender': 'neutral'}
    smpl_tgt.pose[:] = 0
    body_tgt = Mesh(smpl_tgt.r, smpl_tgt.f)

    ## Re-target
    ret = retarget(garment, body_src, body_tgt)

    ## Re-pose
    ret_posed = pose_garment(ret, vert_inds, tgt_params, dpghsmd)
    body_tgt_posed = pose_garment(body_tgt, range(len(body_tgt.v)), tgt_params, dpghsmd)

    ## Remove intersections
    ret_posed_interp = remove_interpenetration_fast(ret_posed, body_tgt_posed)
    ret_posed_interp.vt = garment.vt
    ret_posed_interp.ft = garment.ft
    ret_posed_interp.set_texture_image(garment_tex)

    return ret_posed_interp

path = '../../Multi-Garment_dataset/'
all_scans = glob(path + '*')
garment_classes = ['Pants', 'ShortPants', 'ShirtNoCoat', 'TShirtNoCoat', 'LongCoat']
gar_dict = {}
for gar in garment_classes:
    gar_dict[gar] = glob(join(path, '*', gar + '.obj'))

if __name__ == '__main__':
    dp = SmplPaths()
    vt, ft = dp.get_vt_ft_hres()
    smpl = Smpl(dp.get_hres_smpl_model_data())

    ## This file contains correspondances between garment vertices and smpl body
    fts_file = 'assets/garment_fts.pkl'
    vert_indices, fts = pkl.load(open(fts_file, 'rb') , encoding='latin1')
    fts['naked'] = ft

    ## Choose any garmet type as source
    garment_type = 'TShirtNoCoat'
    index = np.random.randint(0, len(gar_dict[garment_type]))   ## Randomly pick from the digital wardrobe
    path = split(gar_dict[garment_type][index])[0]


    garment_org_body_unposed = load_smpl_from_file(join(path, 'registration.pkl'))
    garment_org_body_unposed.pose[:] = 0
    garment_org_body_unposed.trans[:] = 0
    garment_org_body_unposed = Mesh(garment_org_body_unposed.v, garment_org_body_unposed.f)

    garment_unposed = Mesh(filename=join(path, garment_type + '.obj'))
    garment_tex = join(path, 'multi_tex.jpg')

    ## Generate random SMPL body (Feel free to set up ur own smpl) as target subject
    #smpl.pose[:] = np.random.randn(72) *0.05
    smpl.pose[:] = [ 2.91917896e+00,  4.58017327e-02,  1.53551832e-01, 
        -9.22530651e-01,  8.30429792e-02,  3.82127285e-01,
        -8.67569566e-01, -9.41842701e-03, -3.08711380e-01,
         5.51593065e-01, -2.62972210e-02,  1.17724165e-02,
         1.55291820e+00,  9.38382372e-02, -2.03269675e-01,
         1.60624826e+00, -1.72855295e-02,  1.70969218e-01,
        -5.60724996e-02, -2.51412056e-02, -6.79262029e-03,
        -1.70694739e-01,  7.73771927e-02, -6.54179528e-02,
        -2.13481039e-01, -2.78314129e-02, -1.45036180e-03,
        -3.69013101e-02, -1.13427769e-02, -8.66152346e-03,
        -2.59527802e-01,  6.41327873e-02,  1.68739066e-01,
        -2.42282629e-01, -1.39641911e-01, -1.29240543e-01,
        -3.32024634e-01, -9.92928073e-02, -7.45689273e-02,
         5.03234453e-02, -2.15491727e-01, -3.47436190e-01,
         3.79543081e-02,  1.83946118e-01,  3.41378152e-01,
        -4.32795621e-02, -8.62272084e-02,  1.32593354e-02,
         9.13528800e-02, -3.66373211e-01, -8.13994765e-01,
         8.22413862e-02,  2.69980103e-01,  8.60714853e-01,
         2.12322161e-01, -9.02887702e-01,  2.42460236e-01,
         1.48995027e-01,  7.73568034e-01, -1.55371800e-01,
         2.44113714e-01, -3.91888991e-02,  2.08952576e-01,
         1.52510270e-01,  3.13121825e-02, -1.60491675e-01,
        -2.35091552e-01, -1.13584720e-01, -2.44322896e-01,
        -2.06727579e-01,  1.06525600e-01,  2.20032185e-01]
    smpl.betas[:] = np.random.randn(10) *0.01
    smpl.trans[:] = 0
    tgt_body = Mesh(smpl.r, smpl.f)

    vert_inds = vert_indices[garment_type]
    garment_unposed.set_texture_image(garment_tex)

    new_garment = dress(smpl, garment_org_body_unposed, garment_unposed, vert_inds, garment_tex)

    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([garment_unposed]); mvs[0][0].save_snapshot('garment_unposed.png', blocking=True)
    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([new_garment]); mvs[0][0].save_snapshot('new_garment.png', blocking=True)
    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([tgt_body]); mvs[0][0].save_snapshot('tgt_body.png', blocking=True)
    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([garment_org_body_unposed]); mvs[0][0].save_snapshot('garment_org_body_unposed.png', blocking=True)
    mvs = MeshViewers((1, 1))
    mvs[0][0].set_static_meshes([new_garment, tgt_body]); mvs[0][0].save_snapshot('new_garment_tgt_body.png', blocking=True)

    print('Done')

