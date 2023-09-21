import os
import sys
from matplotlib.pyplot import axis
import numpy as np
from pyparsing import White
from raylib import *
import pyray as pr
from DataGeneration import DataGeneration
import glow.Experiment_utils as exputils


def initViewerSetting(screenWidth, screenHeight, targetFPS):
    InitWindow(screenWidth, screenHeight, b"raylib [models] example - data augmentation")
    #camera = ffi.new("struct Camera3D *", [[18.0, 16.0, 18.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 45.0, 0])

    # Define the camera to look into our 3d world
    camera = pr.Camera3D([0])
    camera.position = pr.Vector3(0.0, 10.0, 10.0)   # Camera position
    camera.target = pr.Vector3(0.0, 0.0, 0.0)       # Camera looking at point
    camera.up = pr.Vector3(0.0, 1.0, 0.0)           # Camera up vector (rotation towards target)
    camera.fovy = 45.0                                 # Camera field-of-view Y
    camera.projection = pr.CAMERA_PERSPECTIVE # CAMERA_PERSPECTIVE       # Camera mode type

    #SetCameraMode(camera[0], CAMERA_ORTHOGRAPHIC)
    SetTargetFPS(targetFPS)  # Set our game to run at 60 frames-per-second
    return camera, screenWidth, screenHeight, targetFPS

def setGroundSetting(width,height,resw,resh,shader_vs,shader_fs):
    ground_plane_shader = LoadShader(shader_vs, shader_fs);
    ground_plane_mesh = GenMeshPlane(width, height, resw, resh);
    ground_plane_model = LoadModelFromMesh(ground_plane_mesh);
    ground_plane_model.materials[0].shader = ground_plane_shader;

    return ground_plane_model

bool_go = False
bool_stop = True
camera, screenWidth, screenHeight, targetFPS = initViewerSetting(1920,1080,60)
ground_plane_model = setGroundSetting(20,20,100,100,b"D:/TJ_develop/Python_develop/raylib-python-cffi/checkerboard.vs", b"D:/TJ_develop/Python_develop/raylib-python-cffi/checkerboard.fs")
DisableCursor();                    # Limit cursor to relative movement inside the window


"""
Init

"""
scene_dg = DataGeneration()
bool_predict = False

nBatch = 0  
nGlobalCnt = 0
nGlobalMaxCnt = 70
nBatchMaxCnt = 30

target_isMat = False

bool_load = False
"""
Update 

"""
#update Function
while(bool_stop):
    # Update
    #----------------------------------------------------------------------------------
    #UpdateCamera(camera,CAMERA_FREE)              # Update camera
    #pr.begin_mode_3d(camera)
    pr.update_camera(camera,pr.CAMERA_THIRD_PERSON)
    # Event (TODO)
    #----------------------------------------------------------------------------------        
    if IsKeyDown(KEY_Y):
        bool_go = not bool_go
        
    if IsKeyDown(KEY_ESCAPE):
        bool_stop = False
    
    if IsKeyDown(KEY_K):
        target_isMat = not target_isMat

    """ Generating Training Data in text files in the folder """
    if IsKeyDown(KEY_T):
        scene_dg.gen_learning_data_fromTXT("D:/TJ_develop/2022/Export/2022/Train",bool_save=True,window_step=40,Train=True,target_isMat=target_isMat)
    
    """ Generating Test Data in text files in the folder """
    if IsKeyDown(KEY_P):
        scene_dg.gen_learning_data_fromTXT("D:/TJ_develop/2022/Export/2022/Test",bool_save=True,window=600,window_step=600,Train=True,target_isMat=target_isMat)
    
    """ Visualize World Positions using Scaler"""
    if IsKeyDown(KEY_S):
        learningdatas = np.load(f'D:/TJ_develop/2022/Export/2022/Train/npz_position/Train_sliced.npz')['clips'].astype(np.float32)

        if(target_isMat):
            scalingdatas = np.concatenate((
            learningdatas[...,:66],
            learningdatas[...,-(9+4):-4]), axis=-1).copy()
        else:
            scalingdatas = np.concatenate((
                        learningdatas[...,:66],
                        learningdatas[...,-(3+4):-4]), axis=-1).copy()
         
        # update scaler 
        scene_dg.scaler = scene_dg.gen_scaler(scalingdatas,False,"D:/TJ_develop/2022/Export/Train/npz_position")

        if(target_isMat):
            scaled_pose = exputils.Normalize_motion(scalingdatas[...,:66],scene_dg.scaler)
            scaled_vel = exputils.Normalize_tar_root(scalingdatas[...,-9:],scene_dg.scaler)

            unscaled_pose = exputils.unNormalize_motion(scaled_pose,scene_dg.scaler)
            unscaled_vel = exputils.unNormalize_tar_root(scaled_vel,scene_dg.scaler)

            joints,translations = scene_dg.gen_world_targetdata(unscaled_pose[nBatch],unscaled_vel[nBatch])
        else:
            scaled_pose = exputils.Normalize_motion(scalingdatas[...,:66],scene_dg.scaler)
            scaled_vel = exputils.Normalize_vel(scalingdatas[...,-3:],scene_dg.scaler)

            unscaled_pose = exputils.unNormalize_motion(scaled_pose,scene_dg.scaler)
            unscaled_vel = exputils.unNormalize_vel(scaled_vel,scene_dg.scaler)

            joints,translations = scene_dg.gen_world_pos_data(unscaled_pose[nBatch],unscaled_vel[nBatch])
            

        """ drawing init """
        nGlobalMaxCnt = joints.shape[0]
        scene_dg.parents = np.array([0,
            1,2,3,4, 
            1,6,7,8,
            1,10,11,
            12,13,14,15,
            12,17,
            12,19,20,21]) - 1
        bool_load= True
        
    """ Visualize Train & Test Data """
    if IsKeyDown(KEY_O):
        learningdatas = np.load(f'D:/TJ_develop/2022/Export/2022/Test/npz_position/Test_sliced.npz')['clips'].astype(np.float32)
        
        nBatchMaxCnt = learningdatas.shape[0]
        
        if(target_isMat):
            joints,translations = scene_dg.gen_world_targetdata(learningdatas[nBatchMaxCnt-1,:,:66],learningdatas[nBatchMaxCnt-1,:,-(9+4):-(4)])
        else:
            joints,translations = scene_dg.gen_world_pos_data(learningdatas[nBatchMaxCnt-1,:,:66],learningdatas[nBatchMaxCnt-1,:,-(3+4):-(4)])

        nGlobalMaxCnt = joints.shape[0]
        scene_dg.parents = np.array([0,
            1,2,3,4, 
            1,6,7,8,
            1,10,11,
            12,13,14,15,
            12,17,
            12,19,20,21]) - 1
        bool_load = True

    
    if IsKeyDown(KEY_R):
        learningdatas = np.load(f'D:/TJ_develop/2022/Export/2022/Train/npz_position/Train_sliced.npz')['clips'].astype(np.float32)
        
        nBatchMaxCnt = learningdatas.shape[0]
        
        if(target_isMat):
            joints,translations = scene_dg.gen_world_targetdata(learningdatas[nBatchMaxCnt-1,:,:66],learningdatas[nBatchMaxCnt-1,:,-(9+4):-(4)])
        else:
            joints,translations = scene_dg.gen_world_pos_data(learningdatas[nBatchMaxCnt-1,:,:66],learningdatas[nBatchMaxCnt-1,:,-(3+4):-(4)])
        
        nGlobalMaxCnt = joints.shape[0]
        scene_dg.parents = np.array([0,
            1,2,3,4, 
            1,6,7,8,
            1,10,11,
            12,13,14,15,
            12,17,
            12,19,20,21]) - 1
        bool_load = True
    
    #----------------------------------------------------------------------------------
    BeginDrawing()

    # animation roop
    if(nGlobalCnt >= nGlobalMaxCnt) or (nGlobalCnt < 0):
        nGlobalCnt = 0
    
    ClearBackground(RAYWHITE)

    BeginMode3D(camera)

    DrawModel(ground_plane_model, (0.0, -0.01, 0.0), 10.0, WHITE)
    """
    draw joints
    """
    scale =1.00
    
    if(bool_load):
        scene_dg.update_drawJoints(nGlobalCnt,joints*scale,RED)

        #scene_dg.update_drawTrajectory(translations*scale,GREEN)
    
    #scene_dg.update_drawEnvs(nGlobalCnt,centerpos*scale,occupancy,GREEN)

    #scene_dg.update_drawToTarget(trajectory,TargetPositions,BLUE)
    

    DrawGrid(20, 1.0)
    
    EndMode3D()

    # GUI (TODO)
    #----------------------------------------------------------------------------------
    # pr.gui_text_box( pr.Rectangle(100, 90 + 50, 200, 50 ),f"Train_{train_X[nBatch,nGlobalCnt,-1]}_{train_X.shape[0]}", 50,False)
    # pr.gui_text_box( pr.Rectangle(100, 90 + 100, 200, 50 ),f"Valid_{valid_X[nBatch,nGlobalCnt,-1]}_{valid_X.shape[0]}", 50,False)
    # pr.gui_text_box( pr.Rectangle(100, 90 + 150, 200, 50 ),f"Test_{test_X[nBatch,nGlobalCnt,-1]}_{test_X.shape[0]}", 50,False)
    # pr.gui_text_box( pr.Rectangle(100, 90 + 100, 200, 50 ),f"Contact_L_{foot_fl[nGlobalCnt]}", 90,False)
    # pr.gui_text_box( pr.Rectangle(100, 90 + 170, 200, 50 ),f"Contact_L_{foot_tl[nGlobalCnt]}", 90,False)
    
    # pr.gui_text_box( pr.Rectangle(100, 90 + 240, 200, 50 ),f"Contact_R_{foot_fr[nGlobalCnt]}", 90,False)
    # pr.gui_text_box( pr.Rectangle(100, 90 + 310, 200, 50 ),f"Contact_R_{foot_tr[nGlobalCnt]}", 90,False)

    pr.gui_text_box( pr.Rectangle(100, 90 + 240, 200, 50 ),f"target_is_Mat? : {target_isMat}", 90,False)

    nGlobalCnt = pr.gui_slider_bar(
        pr.Rectangle(100, screenHeight - 100, 500, 70 ),
        "nFrame", 
        f"nFrame_{str(nGlobalCnt)}",
        nGlobalCnt,
        0, nGlobalMaxCnt - 1)
    nGlobalCnt = int(nGlobalCnt)

    nBatch = pr.gui_slider_bar(
        pr.Rectangle(100, 90 + 10, 500, 70 ), 
        "nBatch", 
        f"nFrame_{str(nBatch)}",
        nBatch,
        0, nBatchMaxCnt - 1)
    nBatch = int(nBatch)
    #----------------------------------------------------------------------------------
    DrawFPS(10, 10)

    if (bool_go == True):
        nGlobalCnt = nGlobalCnt +1

    EndDrawing()
# De-Initialization
#--------------------------------------------------------------------------------------
UnloadModel(ground_plane_model)         # Unload model

CloseWindow()              # Close window and OpenGL context