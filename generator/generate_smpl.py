import torch
import cv2
import os
import numpy as np
from bodymocap.models import SMPL
from renderer.visualizer import Visualizer
import mocap_utils.demo_utils as demo_utils
from mocap_utils.coordconv import convert_bbox_to_oriIm, convert_smpl_to_bbox
from renderer import meshRenderer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def generateSMPLImage(out_img_size, img_original_bgr, generated_mesh_list ):
    #renderer = 'opengl'
    renderer = meshRenderer.meshRenderer()
    renderer.setRenderMode('geo')
    renderer.offscreenMode(True)

    renderer.setWindowSize(out_img_size, out_img_size)
    renderer.setBackgroundTexture(img_original_bgr)
    renderer.setViewportSize(out_img_size,out_img_size)

    # self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
    renderer.clear_mesh()

    generated_mesh_list_offset = generated_mesh_list[0]['vertices'].copy()
    generated_mesh_list_offset[:,0] -= out_img_size*0.5
    generated_mesh_list_offset[:,1] -= out_img_size*0.5
    render_mesh_list = []
    render_mesh_list.append( {'ver':generated_mesh_list_offset,
                            'f':generated_mesh_list[0]['faces']})

    #for mesh in meshList:
    renderer.add_mesh(render_mesh_list[0]['ver'],
                    render_mesh_list[0]['f'])
    renderer.showBackground(True)
    renderer.setWorldCenterBySceneCenter()
    renderer.setCameraViewMode("cam")
    # self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])

    renderer.display()
    renderImg = renderer.get_screen_color_ibgr()

    return renderImg

smpl_dir = '/home/ryan/iprobe/frankmocap/extra_data/smpl'
smplModelPath = smpl_dir +  '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
smpl_model = SMPL(smplModelPath, batch_size=1, create_transl=False).to(device)

out_img_height = 1024
out_img_width = 1024

# load pre-trained neural network
#SMPL_MEAN_PARAMS = './extra_data/body_module/data_from_spin/smpl_mean_params.npz'

#generated_betas = torch.randn( (1,10) ).cuda()
#generated_body_pose = torch.rand((1,69)).cuda()
generated_betas=torch.tensor([[-0.0346,  1.2569,  0.5877,  0.7865, -0.4818, -0.0496, -0.4249,  0.1432,
          0.5597,  0.0414]], device='cuda:0')
generated_body_pose=torch.tensor([[-2.7351e-01, -1.9704e-02,  2.7587e-02, -4.0531e-01,  6.2722e-02,
          1.5225e-02,  2.0527e-01, -2.5813e-02, -2.1694e-05,  5.3492e-01,
         -3.8525e-02, -4.5016e-02,  5.2484e-01,  3.7909e-02,  5.3713e-02,
         -4.3767e-02,  3.5737e-03, -1.4514e-02, -2.9318e-02,  1.8223e-01,
         -1.6615e-01, -3.2627e-02, -1.7103e-01,  1.4308e-01,  4.5918e-02,
         -7.7641e-04, -2.1128e-03, -1.7847e-01,  4.0801e-04,  2.6563e-01,
         -1.7916e-01, -3.5204e-02, -2.6819e-01, -9.5199e-02, -8.6418e-02,
          2.8292e-03, -1.3506e-02, -3.3374e-01, -1.9740e-01, -4.2988e-02,
          3.6691e-01,  1.4963e-01,  7.1456e-02, -5.8325e-02,  3.0682e-02,
          1.4356e-01, -3.6185e-01, -7.9421e-01,  6.0814e-02,  4.3248e-01,
          7.5941e-01,  1.0132e-01, -1.2516e+00,  3.0230e-01,  1.0291e-01,
          1.2303e+00, -2.8050e-01, -4.3411e-02, -1.0835e-01,  1.7673e-01,
         -5.9026e-02,  1.1500e-01, -1.6580e-01, -1.2416e-01, -8.6635e-02,
         -1.8823e-01, -1.0441e-01,  8.1194e-02,  1.7734e-01]], device='cuda:0')
generated_global_orient=torch.tensor([[ 2.9150,  0.1813, -0.9472]], device='cuda:0')
#generated_global_orient=torch.tensor([[ 0.0, 0.0, 0.0]], device='cuda:0')

#generated_global_orient = torch.rand((1,3)).cuda()
#generated_camera = torch.rand((1,3)).cuda()
generated_camera = torch.tensor([[1.0, 0.0, 0.0]], device='cuda:0')

smpl_output = smpl_model(
    betas=generated_betas,
    body_pose=generated_body_pose,
    global_orient=generated_global_orient,
    pose2rot=True)

generated_vertices = smpl_output.vertices
generated_joints_3d = smpl_output.joints
generated_vertices = generated_vertices[0].cpu().numpy()

generated_camera = generated_camera.cpu().numpy().ravel()
camScale = generated_camera[0]
camTrans = generated_camera[1:]

visualizer = Visualizer("opengl")

# generated_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
# center the bounding box
bbox_height = 100
bbox_width = 100
top = (out_img_height-bbox_height)/2
left = (out_img_width-bbox_width)/2

gen_vertices = smpl_output.vertices
gen_vertices = gen_vertices[0].cpu().numpy()
gen_vertices_bbox = convert_smpl_to_bbox(gen_vertices,generated_camera[0],
                                         generated_camera[1:])
vertices = convert_bbox_to_oriIm(
    gen_vertices_bbox, 0.8, (top,left,), out_img_width, out_img_height)
generated_mesh_list = [ dict(vertices=vertices, faces=smpl_model.faces) ]

img_original_bgr = np.zeros( (out_img_width,out_img_height,3), np.uint8 )


dummy_bbox = [top,left,bbox_height,bbox_width]
res_img = visualizer.visualize(
    img_original_bgr,
    pred_mesh_list=generated_mesh_list,
    body_bbox_list=[dummy_bbox]
)

out_dir = "/home/ryan/iprobe/frankmocap/temp_smpl_images/"

cv2.imwrite( os.path.join( out_dir, 'test_image.jpg' ), res_img )

'''
#renderer = 'opengl'
renderer = meshRenderer.meshRenderer()
renderer.setRenderMode('geo')
renderer.offscreenMode(True)

renderer.setWindowSize(out_img_size, out_img_size)
renderer.setBackgroundTexture(img_original_bgr)
renderer.setViewportSize(out_img_size,out_img_size)

# self.renderer.add_mesh(meshList[0]['ver'],meshList[0]['f'])
renderer.clear_mesh()

generated_mesh_list_offset = generated_mesh_list[0]['vertices'].copy()
generated_mesh_list_offset[:,0] -= 1024*0.5
generated_mesh_list_offset[:,1] -= 1024*0.5
render_mesh_list = []
render_mesh_list.append( {'ver':generated_mesh_list_offset,
                          'f':generated_mesh_list[0]['faces']})

#for mesh in meshList:
renderer.add_mesh(render_mesh_list[0]['ver'],
                  render_mesh_list[0]['f'])
renderer.showBackground(True)
renderer.setWorldCenterBySceneCenter()
renderer.setCameraViewMode("cam")
# self.renderer.setViewportSize(img_original_resized.shape[1], img_original_resized.shape[0])

renderer.display()
renderImg = renderer.get_screen_color_ibgr()
'''
renderImg = generateSMPLImage( out_img_height, img_original_bgr,
                              generated_mesh_list)

cv2.imwrite( os.path.join( out_dir, 'renderImg.png' ), renderImg )


