from turtle import position
from anyio import create_semaphore
import numpy as np
from numpy.lib.function_base import blackman
from raylib import colors
from raylib.colors import BLACK, BLUE, BROWN, DARKBLUE, GOLD, GREEN, PURPLE
from scipy.spatial.transform import Rotation as R
import os
from raylib import *
import pyray as pr
import sys

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#sys.path.append('holdens')
# import BVH as BVH
# import Animation as Animation
# from Quaternions import Quaternions
# from Pivots import Pivots
import scipy.ndimage.filters as filters

"""
input: MotionData (npz)
output: Scaler, Trainable Data (x, condition)

"""
class DataGeneration():

    def __init__(self):
        
        self.translations = np.zeros((80,3))
        self.translations_new = np.zeros((8,80,3))
        self.joints = np.zeros((80,22,3))
        self.joints_new = np.zeros((8,80,22,3))

        self.parents = np.array([0,1,2,3,4,5, 
            4,7,8,9,
            4,11,12,13,
            1,15,16,17,
            1,19,20,21]) - 1
        
        self.parents_ee = np.array([5, 
            9,
            13,
            17,
            21]) - 1

        self.data_X = []
        self.batch_X = []
        
        self.scaler = StandardScaler()

        self.joints = []

        self.colors_8 = [PURPLE,GREEN,BLUE,BROWN,
        DARKBLUE,GOLD,colors.GRAY,colors.MAROON]
    """
    load files
    """
    def get_npzfiles(self,directory):
        return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
        if os.path.isfile(os.path.join(directory,f))
        and f.endswith('.npz')] 
    
    def get_bvhfiles(self,directory):
        return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
        if os.path.isfile(os.path.join(directory,f))
        and f.endswith('.bvh') and f != 'rest.bvh'] 

    def get_txtfiles(self,directory):
        return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
        if os.path.isfile(os.path.join(directory,f))
        and f.endswith('.txt')] 

    def process_file_0530_txt(self,filename, window=80, window_step=40, divide = 1, filterwidth=20, label=None):
    
        clips = np.loadtxt(filename, delimiter=" ")
        
        datas = clips.copy()

        """ position """
        positions = datas[:,:66]
        """ rotation """
        rotations = datas[:,66:132]
        """ end-effector position """
        ee_positions = datas[:,132:132+5*3]
        """ end-effector rotation """
        ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
        """ environment """
        env_centerpos = datas[:,-(2640*4+ (3*2)):-(2640+(3*2))]
        env_occupancy = datas[:,-(2640 + (3*2)):-(3*2)]
        """ root velocity """
        root_positions = datas[:,-(3*2):-(3)]
        root_forwards = datas[:,-(3):]
        # linear
        velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
        # rotation
        target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
        rotation = Quaternions.between(root_forwards, target)#[:,np.newaxis]    
        #calc
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        #first velocity is zero 
        zero_vel = np.array([[0,0,0]])
        velocity = np.concatenate([zero_vel,velocity], axis=0)
        rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 
        """ gen testdata"""
        draw_data = np.concatenate((positions,rotations,ee_positions,ee_rotations,env_centerpos,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
        
        #positions = np.concatenate((positions,rotations,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1)
        """ gen traindata"""
        positions = np.concatenate((positions[:,:3],rotations,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1)
        if (label is not None):
            label_arr = np.zeros((positions.shape[0],1))
            label_arr[:,0] = label
            positions = np.concatenate([positions, label_arr], axis=-1)


        """ Slide over windows """
        windows = []
        windows_classes = []
        label_id = 0
        if (label is not None): 
            label_id = 1
        for j in range(0, len(positions)-window//8, window_step):
        
            """ If slice too small pad out by repeating start and end poses """
            slice = positions[j:j+window]
            if len(slice) < window:
                left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                left[:,-(3+label_id):-(label_id)] = 0.0
                right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                right[:,-(3+label_id):-(label_id)] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            
            if len(slice) != window: raise Exception()
            
            windows.append(slice)
            
            """ Find Class """
            cls = -1
            windows_classes.append(cls)
            
        return windows, draw_data, windows_classes

    def process_file_fromUnity(self,filename, filterwidth=20, label=None, b_load= False):
        if(b_load):
            datas = np.load(os.path.join(filename,'SeqDemo.npz'))['clips'].astype(np.float32)
        else:
            datas = np.loadtxt(os.path.join(filename,'SeqDemo.txt'), delimiter=" ")
            np.savez_compressed(os.path.join(filename,'SeqDemo.npz'), clips = datas)
            datas = np.load(os.path.join(filename,'SeqDemo.npz'))['clips'].astype(np.float32)
    
        start = 1
        upper_condition = datas[:,:1]
        #print(index)
        positions = datas[:,start:start+66].copy()
        env_occupancy = datas[:,start+66:start+66+2640].copy()
        
        velocity = datas[:,-3:].copy()
        if(filterwidth == 0):
            rvelocity = velocity[:,-1]
            lvelocity_x = velocity[:,0]
            lvelocity_z = velocity[:,1]
        else:
            rvelocity = filters.gaussian_filter1d(velocity[:,-1], filterwidth, axis=0, mode='nearest')  
            lvelocity_x = filters.gaussian_filter1d(velocity[:,0], 3, axis=0, mode='nearest')  
            lvelocity_z = filters.gaussian_filter1d(velocity[:,1], 3, axis=0, mode='nearest')

        filtered_velocity = datas[:,-3:].copy()
        filtered_velocity[:,0] = lvelocity_x
        filtered_velocity[:,1] = lvelocity_z
        filtered_velocity[:,2] = rvelocity

        """ gen traindata"""
        ee_pos = np.zeros((positions.shape[0],30))
        positions = np.concatenate((positions,ee_pos,env_occupancy,filtered_velocity), axis=-1).copy()
        #positions = np.concatenate((positions,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1).copy()
        if (label is not None):
            label_arr = np.zeros((positions.shape[0],1))
            label_arr[:,0] = label
            positions = np.concatenate([positions, label_arr], axis=-1)
        
        return positions , upper_condition
        


    def process_root_vel(self, clips , vels, filename=None):
        if filename is not None:
            clips = np.loadtxt(filename, delimiter=" ")
        else:
            datas = clips.copy()
        
        nFrame, _ = datas.shape
        nMirror = nFrame // 2 
        
        
        total_targetVels = datas[:,:3].copy()

        """target velocity"""
        joints, trajectory = self.gen_world_pos_data(datas[:nMirror,:66],vels)
        

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        targetVels = np.zeros((joints.shape[0],3))
        
        root_dx, root_dz, root_dr = vels[...,-3], vels[...,-2], vels[...,-1]
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            targetvel = rot.inv().apply(trajectory[-1] - translation) # velocity 로 이동하기 전 frame 으로 표현한 목표로의 velocity
            targetVels[i,:] = targetvel

            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]])) # velocity 만큼 translation 함
            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot # velocity 만큼 rotation 함

        total_targetVels[:nMirror] = targetVels

        """target velocity"""
        joints, trajectory = self.gen_world_pos_data(datas[nMirror:,:66],vels)
        

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        targetVels = np.zeros((joints.shape[0],3))
        
        root_dx, root_dz, root_dr = vels[...,-3], vels[...,-2], vels[...,-1]
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            targetvel = rot.inv().apply(trajectory[-1] - translation) # velocity 로 이동하기 전 frame 으로 표현한 목표로의 velocity
            targetVels[i,:] = targetvel

            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]])) # velocity 만큼 translation 함
            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot # velocity 만큼 rotation 함

        total_targetVels[nMirror:] = targetVels

        return total_targetVels

    def process_foot_contact(self, clips , vels, threshold=0.02, filename=None):
        if filename is not None:
            clips = np.loadtxt(filename, delimiter=" ")
        else:
            datas = clips.copy()
        
        joints, trajectory = self.gen_world_pos_data(datas[:,:66],vels)

        """footvel"""
        joints_sample =joints[:,:66].copy()
        joints_footcontact_fr = np.zeros((joints_sample.shape[0]-1,1))
        joints_footcontact_fl = np.zeros((joints_sample.shape[0]-1,1))
        joints_footcontact_tr = np.zeros((joints_sample.shape[0]-1,1))
        joints_footcontact_tl = np.zeros((joints_sample.shape[0]-1,1))

        joints_sample = joints_sample.reshape((len(joints_sample), -1, 3))
        joints_vel_fr = np.linalg.norm(joints_sample[:-1,16] - joints_sample[1:,16],axis=-1) # right foot
        joints_vel_fl = np.linalg.norm(joints_sample[:-1,20] - joints_sample[1:,20],axis=-1) # left foot
        joints_vel_tr = np.linalg.norm(joints_sample[:-1,17] - joints_sample[1:,17],axis=-1) # right toe
        joints_vel_tl = np.linalg.norm(joints_sample[:-1,21] - joints_sample[1:,21],axis=-1) # left toe

        #joints_foot_contact = np.where(joints_vel_r<0.004)
        joints_footcontact_fr[joints_vel_fr<threshold] = 1
        joints_footcontact_fl[joints_vel_fl<threshold] = 1

        joints_footcontact_tr[joints_vel_tr<threshold] = 1
        joints_footcontact_tl[joints_vel_tl<threshold] = 1


        joints_footcontact_fr = np.concatenate((joints_footcontact_fr,np.ones((1,1))),axis=0)
        joints_footcontact_fl = np.concatenate((joints_footcontact_fl,np.ones((1,1))),axis=0)

        joints_footcontact_tr = np.concatenate((joints_footcontact_tr,np.ones((1,1))),axis=0)
        joints_footcontact_tl = np.concatenate((joints_footcontact_tl,np.ones((1,1))),axis=0)


        return joints_footcontact_fr, joints_footcontact_fl, joints_footcontact_tr, joints_footcontact_tl 

    def trim_AB(self,positions,nHalfData):
        """ triming for desired estimation """
        positions_A = positions[:nHalfData].copy() # \tau +1
        positions_A = positions_A[1:]
        positions_B = positions[nHalfData:].copy()
        positions_B = positions_B[1:]
        positions_cur = np.concatenate((positions_A,positions_B),axis=0)
        return positions_cur

    def process_file_2023_txt_position(self,filename, window=81, window_step=40, Train=True, b_load= False):
        
            if(b_load):
                datas = np.load(os.path.join(filename,'Input.npz'))['clips'].astype(np.float32)
            else:
                datas = np.loadtxt(os.path.join(filename,'Input.txt'), delimiter=" ")
                
                np.savez_compressed(os.path.join(filename,'Input.npz'), clips = datas)
                datas = np.load(os.join(filename,'Input.npz'))['clips'].astype(np.float32)
            
            
            """ position """
            start = 0
            positions = datas[:,start:start + 66]
            
            """ rotation """
            start = start + 66
            rotations = datas[:,start:start + 22*6]

            """ end-effector position """
            start = start + 22*6
            ee_positions = datas[:,start:start+5*3]

            """ end-effector rotation """
            start = start + 5*3
            ee_rotations = datas[:,start:start + 5*6]

            """ environment """
            start = start + 5*6
            env_centerpos = datas[:,start:start+(2640*3)]

            start = start + 2640*3
            env_occupancy = datas[:,start:start + 2640]
            
            """ target root """
            start = start + 2640
            target_root = datas[:,start:start+9]

            """ foot contact """
            start = start+9
            pre_foot_contact = datas[:,start:start+2]
            
            start = start+2
            cur_foot_contact = datas[:,start:start+2]

            start = start+2
            print(f"{datas.shape[0]}/data_{str(start)}")
            
            
            
            """ gen learning data """
            learningdata = np.concatenate((positions,ee_positions,
            env_occupancy,
            target_root,
            pre_foot_contact,
            cur_foot_contact), axis=-1).copy()

            """draw data"""
            draw_data = np.concatenate((
                positions,
                ee_positions,
                env_centerpos,
                env_occupancy,
                target_root,
                pre_foot_contact,
                cur_foot_contact
            ), axis=-1).copy()

            """ Slide over windows """
            windows = []
            windows_drawdata = []
            #-window//8
            if(Train):
                for j in range(0, len(learningdata), window_step):
                
                    """ If slice too small pad out by repeating start and end poses """
                    slice = learningdata[j:j+window]
                    sliced_drawdata = draw_data[j:j+window]

                    if len(slice) < window:
                        left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                        left[:,-(9+4):-(4+6)] = 0.0 # local position is stopped
                        right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                        right[:,-(9+4):-(4+6)] = 0.0 # local position is stopped
                        slice = np.concatenate([left, slice, right], axis=0)

                        left_draw  = sliced_drawdata[:1].repeat((window-len(sliced_drawdata))//2 + (window-len(sliced_drawdata))%2, axis=0)
                        left_draw[:,-(9+4):-(4+6)] = 0.0 # local position is stopped
                        right_draw = sliced_drawdata[-1:].repeat((window-len(sliced_drawdata))//2, axis=0)
                        right_draw[:,-(9+4):-(4+6)] = 0.0 # local position is stopped
                        sliced_drawdata = np.concatenate([left_draw, sliced_drawdata, right_draw], axis=0)

                    
                    if len(slice) != window: raise Exception()
                    
                    windows.append(slice)
                    #windows_drawdata.append(sliced_drawdata)
            else:    
                windows.append(learningdata)
                #windows_drawdata.append(draw_data)

            return windows, windows_drawdata

    def process_file_2023PG_txt_position(self,filename, window=81, window_step=40, Train=True, b_load = False):

            if(b_load):
                datas = np.load(os.path.join(filename,'Input.npz'))['clips'].astype(np.float32)
            else:
                datas = np.loadtxt(os.path.join(filename,'Input.txt'), delimiter=" ")
                
                np.savez_compressed(os.path.join(filename,'Input.npz'), clips = datas)
                datas = np.load(os.path.join(filename,'Input.npz'))['clips'].astype(np.float32)
                
            
            """ position """
            start = 0
            positions = datas[:,start:start + 66]
            
            """ rotation """
            start = start + 66
            rotations = datas[:,start:start + 22*6]

            """ end-effector position """
            start = start + 22*6
            ee_positions = datas[:,start:start+5*3]

            """ end-effector rotation """
            start = start + 5*3
            ee_rotations = datas[:,start:start + 5*6]

            """ environment """
            start = start + 5*6
            env_centerpos = datas[:,start:start+(2640*3)]

            start = start + 2640*3
            env_occupancy = datas[:,start:start + 2640]
            
            """ target root """
            start = start + 2640
            target_root = datas[:,start:start+3]

            """ foot contact """
            start = start+3
            pre_foot_contact = datas[:,start:start+2]
            
            start = start+2
            cur_foot_contact = datas[:,start:start+2]

            start = start+2
            print(f"{datas.shape[0]}/data_{str(start)}")
            
            
            
            """ gen learning data """
            learningdata = np.concatenate((positions,ee_positions,
            env_occupancy,
            target_root,
            pre_foot_contact,
            cur_foot_contact), axis=-1).copy()

            """draw data"""
            draw_data = np.concatenate((
                positions,
                ee_positions,
                env_centerpos,
                env_occupancy,
                target_root,
                pre_foot_contact,
                cur_foot_contact
            ), axis=-1).copy()

            """ Slide over windows """
            windows = []
            windows_drawdata = []
            #-window//8
            if(Train):
                for j in range(0, len(learningdata), window_step):
                
                    """ If slice too small pad out by repeating start and end poses """
                    slice = learningdata[j:j+window]
                    sliced_drawdata = draw_data[j:j+window]

                    if len(slice) < window:
                        left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                        left[:,-(3+4):-(4)] = 0.0 # local position is stopped
                        right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                        right[:,-(3+4):-(4)] = 0.0 # local position is stopped
                        slice = np.concatenate([left, slice, right], axis=0)

                        left_draw  = sliced_drawdata[:1].repeat((window-len(sliced_drawdata))//2 + (window-len(sliced_drawdata))%2, axis=0)
                        left_draw[:,-(3+4):-(4)] = 0.0 # local position is stopped
                        right_draw = sliced_drawdata[-1:].repeat((window-len(sliced_drawdata))//2, axis=0)
                        right_draw[:,-(3+4):-(4)] = 0.0 # local position is stopped
                        sliced_drawdata = np.concatenate([left_draw, sliced_drawdata, right_draw], axis=0)

                    
                    if len(slice) != window: raise Exception()
                    
                    windows.append(slice)
                    #windows_drawdata.append(sliced_drawdata)
            else:    
                windows.append(learningdata)
                #windows_drawdata.append(draw_data)

            return windows, windows_drawdata
    
    
    def process_file_0530_txt_position(self,filename, window=80, window_step=40, divide = 1, filterwidth=20, label=None):
        
            clips = np.loadtxt(filename, delimiter=" ")
            
            datas = clips.copy()

            """ position """
            positions = datas[:,:66].copy()
            """ rotation """
            rotations = datas[:,66:132].copy()
            """ end-effector position """
            ee_positions = datas[:,132:132+5*3].copy()
            """ end-effector rotation """
            ee_rotations = datas[:,(132+5*3):(132+5*3*2)].copy()
            """ environment """
            env_centerpos = datas[:,-(2640*4+ (3)):-(2640+(3))].copy()
            env_occupancy = datas[:,-(2640 + (3)):-(3)].copy()
            
            """ root velocity """
            velocity = datas[:,-3:].copy()
            rvelocity = filters.gaussian_filter1d(velocity[:,-1], filterwidth, axis=0, mode='nearest')  
            lvelocity_x = filters.gaussian_filter1d(velocity[:,0], 3, axis=0, mode='nearest')  
            lvelocity_z = filters.gaussian_filter1d(velocity[:,1], 3, axis=0, mode='nearest')  
            
            filtered_velocity = datas[:,-3:].copy()
            filtered_velocity[:,0] = lvelocity_x
            filtered_velocity[:,1] = lvelocity_z
            filtered_velocity[:,2] = rvelocity
            
            """ foot contact """
            foot_fr, foot_fl, foot_tr, foot_tl = self.process_foot_contact(datas,filtered_velocity,threshold=0.02)
            

            
            nTotalData = datas.shape[0]
            nHalfData = nTotalData // 2 



            """ target """
            targetVels = self.process_root_vel(datas,filtered_velocity)
           
            """ current environment """
            env_A = env_occupancy[:nHalfData].copy() # \tau
            env_A = env_A[:-1]
            env_B = env_occupancy[nHalfData:].copy() # \tau
            env_B = env_B[:-1]
            env_occupancy_prev = np.concatenate((env_A,env_B), axis=0)

            """ triming for desired estimation """
            positions_cur = self.trim_AB(positions,nHalfData)
            ee_positions_cur = self.trim_AB(ee_positions,nHalfData)
            ee_rotations_cur = self.trim_AB(ee_rotations,nHalfData)
            foot_fr_cur = self.trim_AB(foot_fr,nHalfData)
            foot_fl_cur = self.trim_AB(foot_fl,nHalfData)
            foot_tr_cur = self.trim_AB(foot_tr,nHalfData)
            foot_tl_cur = self.trim_AB(foot_tl,nHalfData)
            targetVels_cur = self.trim_AB(targetVels,nHalfData)
            env_centerpos_cur = self.trim_AB(env_centerpos,nHalfData)
            env_occupancy_cur = self.trim_AB(env_occupancy,nHalfData)
            filtered_velocity_cur = self.trim_AB(filtered_velocity,nHalfData)

            # velocity[:,0] = lvelocity_x
            # velocity[:,1] = lvelocity_z
            # velocity[:,-1] = rvelocity

            # """ position """
            # positions = datas[:,:66]
            # """ rotation """
            # rotations = datas[:,66:132]
            # """ end-effector position """
            # ee_positions = datas[:,132:132+5*3]
            # """ end-effector rotation """
            # ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
            # """ environment """
            # env_centerpos = datas[:,-(2640*4+ (3*2)):-(2640+(3*2))]
            # env_occupancy = datas[:,-(2640 + (3*2)):-(3*2)]
            # """ root velocity """
            # root_positions = datas[:,-(3*2):-(3)]
            # root_forwards = datas[:,-(3):]
            # # linear
            # velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
            # # rotation
            # target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
            # rotation = Quaternions.between(root_forwards, target).copy()#[:,np.newaxis]    
            # #calc
            # velocity = rotation[1:] * velocity
            # rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
            # #first velocity is zero 
            # zero_vel = np.array([[0,0,0]])
            # velocity = np.concatenate([zero_vel,velocity], axis=0)
            # rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 


            """ gen testdata"""
            draw_data = np.concatenate((positions,ee_positions,ee_rotations,env_centerpos,env_occupancy,velocity),axis=-1)
            
            #filterd_draw_data = np.concatenate((positions,ee_positions,ee_rotations,foot_fr,foot_fl,foot_tr,foot_tl,targetVels,env_centerpos,env_occupancy,filtered_velocity),axis=-1)
            
            filterd_draw_data = np.concatenate((positions_cur,ee_positions_cur,ee_rotations_cur,
            foot_fr_cur,foot_fl_cur,foot_tr_cur,foot_tl_cur,
            targetVels_cur,env_occupancy_prev,
            env_centerpos_cur,env_occupancy_cur,filtered_velocity_cur),axis=-1)
            
            
            """ gen traindata"""
            #positions = np.concatenate((positions,ee_positions,ee_rotations,foot_fr,foot_fl,foot_tr,foot_tl,targetVels, env_occupancy,filtered_velocity), axis=-1).copy()
            positions = np.concatenate((positions_cur,ee_positions_cur,ee_rotations_cur,
            foot_fr_cur,foot_fl_cur,foot_tr_cur,foot_tl_cur,
            targetVels_cur,env_occupancy_prev,
            env_occupancy_cur,filtered_velocity_cur), axis=-1).copy()
            

            if (label is not None):
                label_arr = np.zeros((positions.shape[0],1))
                label_arr[:,0] = label
                positions = np.concatenate([positions, label_arr], axis=-1)
                

            """ Slide over windows """
            windows = []
            windows_classes = []
            windows_drawdata = []
            label_id = 0
            if (label is not None): 
                label_id = 1
            for j in range(0, len(positions)-window//8, window_step):
            
                """ If slice too small pad out by repeating start and end poses """
                slice = positions[j:j+window]
                sliced_drawdata = filterd_draw_data[j:j+window]

                if len(slice) < window:
                    left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                    left[:,-(3+label_id):-(label_id)] = 0.0
                    right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                    right[:,-(3+label_id):-(label_id)] = 0.0
                    slice = np.concatenate([left, slice, right], axis=0)

                    left_draw  = sliced_drawdata[:1].repeat((window-len(sliced_drawdata))//2 + (window-len(sliced_drawdata))%2, axis=0)
                    left_draw[:,-(3):] = 0.0
                    right_draw = sliced_drawdata[-1:].repeat((window-len(sliced_drawdata))//2, axis=0)
                    right_draw[:,-(3):] = 0.0
                    sliced_drawdata = np.concatenate([left_draw, sliced_drawdata, right_draw], axis=0)

                
                if len(slice) != window: raise Exception()
                
                windows.append(slice)
                windows_drawdata.append(sliced_drawdata)
                """ Find Class """
                cls = -1
                windows_classes.append(cls)
                
            return windows, draw_data, windows_classes, windows_drawdata # draw_data
    def process_file_0530_txt_position_rotation(self,filename, window=80, window_step=40, divide = 1, filterwidth=20, label=None):
    
        clips = np.loadtxt(filename, delimiter=" ")
        
        datas = clips.copy()

        """ position """
        positions = datas[:,:66]
        """ rotation """
        rotations = datas[:,66:132]
        """ end-effector position """
        ee_positions = datas[:,132:132+5*3]
        """ end-effector rotation """
        ee_rotations = datas[:,(132+5*3):(132+5*3*2)]
        """ environment """
        env_centerpos = datas[:,-(2640*4+ (3*2)):-(2640+(3*2))]
        env_occupancy = datas[:,-(2640 + (3*2)):-(3*2)]
        """ root velocity """
        root_positions = datas[:,-(3*2):-(3)]
        root_forwards = datas[:,-(3):]
        # linear
        velocity = (root_positions[1:,:] - root_positions[:-1,:]).copy()
        # rotation
        target = np.array([[0,0,1]]).repeat(len(root_forwards), axis=0)
        rotation = Quaternions.between(root_forwards, target)#[:,np.newaxis]    
        #calc
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        #first velocity is zero 
        zero_vel = np.array([[0,0,0]])
        velocity = np.concatenate([zero_vel,velocity], axis=0)
        rvelocity = np.concatenate([zero_vel[:,0],rvelocity],axis=0)[:,np.newaxis] 
        """ gen testdata"""
        draw_data = np.concatenate((positions,rotations,ee_positions,ee_rotations,env_centerpos,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
        
        #positions = np.concatenate((positions,rotations,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1)
        """ gen traindata"""
        positions = np.concatenate((positions,rotations,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1)
        if (label is not None):
            label_arr = np.zeros((positions.shape[0],1))
            label_arr[:,0] = label
            positions = np.concatenate([positions, label_arr], axis=-1)


        """ Slide over windows """
        windows = []
        windows_classes = []
        label_id = 0
        if (label is not None): 
            label_id = 1
        for j in range(0, len(positions)-window//8, window_step):
        
            """ If slice too small pad out by repeating start and end poses """
            slice = positions[j:j+window]
            if len(slice) < window:
                left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                left[:,-(3+label_id):-(label_id)] = 0.0
                right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                right[:,-(3+label_id):-(label_id)] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            
            if len(slice) != window: raise Exception()
            
            windows.append(slice)
            
            """ Find Class """
            cls = -1
            windows_classes.append(cls)
            
        return windows, draw_data, windows_classes


    def process_file(self,filename, window=80, window_step=40, divide = 1, filterwidth=20, label=None):
    
        anim, names, frametime = BVH.load(filename)
        
        """ Convert to 60 fps """
        anim = anim[::divide]
        
        """ Do FK """
        global_positions = Animation.positions_global(anim)
        
        """ Remove Uneeded Joints """
        positions = global_positions
        
        # """ Put on Floor """
        # fid_l, fid_r = np.array([16,17]), np.array([20,21])
        # foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
        # floor_height = softmin(foot_heights, softness=0.5, axis=0)
        
        #positions[:,:,1] -= floor_height

        """ Add Reference Joint """
        trajectory_filterwidth = filterwidth
        reference = positions[:,0] * np.array([1,0,1])
        reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
        positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
        
        # """ Get Foot Contacts """
        # velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
        
        # feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
        # feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
        # feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
        # feet_l_h = positions[:-1,fid_l,1]
        # feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        
        # feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
        # feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
        # feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
        # feet_r_h = positions[:-1,fid_r,1]
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        
        """ Get Root Velocity """
        velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
        
        """ Remove Translation """
        positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
        positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
        
        """ Get Forward Direction """
        sdr_l, sdr_r, hip_l, hip_r = 6+1, 10+1, 18+1, 14+1
        across1 = positions[:,hip_l] - positions[:,hip_r]
        across0 = positions[:,sdr_l] - positions[:,sdr_r]
        across = across0 + across1
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
        
        direction_filterwidth = filterwidth
        forward = np.cross(across, np.array([[0,1,0]]))
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

        """ Remove Y Rotation """
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        rotation = Quaternions.between(forward, target)[:,np.newaxis]    
        positions = rotation * positions
        
        """ Get Root Rotation """
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        
        """ Add Velocity, RVelocity, Foot Contacts to vector """
        positions = positions[:-1,1:,:]
        positions = positions.reshape(len(positions), -1)
        positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
        positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
        positions = np.concatenate([positions, rvelocity], axis=-1)
        #positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
        if (label is not None):
            label_arr = np.zeros_like((rvelocity))
            label_arr[:,0] = label
            positions = np.concatenate([positions, label_arr], axis=-1)
            
        """ Slide over windows """
        windows = []
        windows_classes = []
        label_id = 0
        if (label is not None):
            label_id = 1
        for j in range(0, len(positions)-window//8, window_step):
        
            """ If slice too small pad out by repeating start and end poses """
            slice = positions[j:j+window]
            if len(slice) < window:
                left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                left[:,-(3+label_id):-(label_id)] = 0.0
                right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                right[:,-(3+label_id):-(label_id)] = 0.0
                slice = np.concatenate([left, slice, right], axis=0)
            
            if len(slice) != window: raise Exception()
            
            windows.append(slice)
            
            """ Find Class """
            cls = -1
            windows_classes.append(cls)
            
        return windows, windows_classes
    
    """
    SAMP
    """
    def process_file_0530_txt_SAMP_position(self,filename_input,filename_output, window=80, window_step=40, divide = 1, filterwidth=20, label=None):
        
            clips_input = np.loadtxt(filename_input, delimiter=" ")
            clips_output = np.loadtxt(filename_output,delimiter=" ")

            # datas = clips_input.copy()
            # datas_out = clips_output.copy()

            # base_input = os.path.basename(filename_input)
            # np.savez_compressed(f'{base_input}_test', clips=datas)
            # base_output = os.path.basename(filename_output)
            # np.savez_compressed(f'{base_output}_test', clips=datas_out)

            datas = np.load("D:/TJ_develop/2022/SAMP/MotionNet/MotionNet/train/Input.npz")['clips'].astype(np.float32)
            datas_out = np.load("D:/TJ_develop/2022/SAMP/MotionNet/MotionNet/train/Output.npz")['clips'].astype(np.float32)
            
            """ position """
            positions_x = datas[:,0:264:12].copy() # 22* 12
            positions_y = datas[:,1:264:12].copy()
            positions_z = datas[:,2:264:12].copy()
            bones = np.zeros_like(datas[:,:66])
            for i in range(0,22):
                bones[:,3*i+0] = positions_x[:,i]
                bones[:,3*i+1] = positions_y[:,i]
                bones[:,3*i+2] = positions_z[:,i]

            """ environment """
            env_centerpos_x = datas[:,647::4].copy() 
            env_centerpos_y = datas[:,648::4].copy()
            env_centerpos_z = datas[:,649::4].copy()
            env_centerpos = np.zeros_like(datas[:,:512*3])
            for i in range(0,512):
                env_centerpos[:,3*i+0] = env_centerpos_x[:,i]
                env_centerpos[:,3*i+1] = env_centerpos_y[:,i]
                env_centerpos[:,3*i+2] = env_centerpos_z[:,i]
            env_occupancy = datas[:,650::4].copy()
            """ root velocity """
            root_pos_dir_input = datas[:,386:388]
            root_pos_dir_output = datas_out[:,386:388]
            root_pos_dir_rotate = root_pos_dir_input[:,0] * root_pos_dir_output[:,1] - root_pos_dir_output[:,0] * root_pos_dir_input[:,1]
            root_rot_velocity = np.arcsin(root_pos_dir_rotate)

            root_linear_velocity = datas_out[:,384:386]

            rvelocity = filters.gaussian_filter1d(root_rot_velocity, filterwidth, axis=0, mode='nearest')  
            lvelocity_x = filters.gaussian_filter1d(root_linear_velocity[:,0], 3, axis=0, mode='nearest')  
            lvelocity_z = filters.gaussian_filter1d(root_linear_velocity[:,1], 3, axis=0, mode='nearest')  
            
            filtered_velocity = datas[:,-3:].copy()
            filtered_velocity[:,0] = lvelocity_x
            filtered_velocity[:,1] = lvelocity_z
            filtered_velocity[:,2] = rvelocity
           
            initial_velocity = np.zeros_like(datas[0:1,-3:])
            filtered_velocity = np.concatenate((initial_velocity,filtered_velocity[:-1]),axis=0)
            """ gen testdata"""
            filterd_draw_data = np.concatenate((bones,env_centerpos,env_occupancy,filtered_velocity),axis=-1)
            #draw_data = np.concatenate((positions,ee_positions,ee_rotations,env_centerpos,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity),axis=-1)
            
            """ gen traindata"""
            positions = np.concatenate((bones,env_occupancy,filtered_velocity), axis=-1).copy()
            #positions = np.concatenate((positions,ee_positions,ee_rotations,env_occupancy,velocity[:,0:1],velocity[:,2:3],rvelocity), axis=-1).copy()
            if (label is not None):
                label_arr = np.zeros((positions.shape[0],1))
                label_arr[:,0] = label
                positions = np.concatenate([positions, label_arr], axis=-1)

            """ Slide over windows """
            windows = []
            windows_classes = []
            label_id = 0
            if (label is not None): 
                label_id = 1
            for j in range(0, len(positions)-window//8, window_step):
            
                """ If slice too small pad out by repeating start and end poses """
                slice = positions[j:j+window]
                if len(slice) < window:
                    left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                    left[:,-(3+label_id):-(label_id)] = 0.0
                    right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                    right[:,-(3+label_id):-(label_id)] = 0.0
                    slice = np.concatenate([left, slice, right], axis=0)
                
                if len(slice) != window: raise Exception()
                
                windows.append(slice)
                
                """ Find Class """
                cls = -1
                windows_classes.append(cls)
                
            return windows, filterd_draw_data, windows_classes, filterd_draw_data # draw_data
    """
    Scailing
    
    """
    def partial_fit(self,data,scaler):
        shape = data.shape
        flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
        scaler.partial_fit(flat)
        
        return scaler
        
    def inv_standardize(self,data, scaler):      
        shape = data.shape
        flat = data.reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.inverse_transform(flat).reshape(shape)
        return scaled        

    def fit_and_standardize(self,data):
        shape = data.shape
        flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
        scaler = StandardScaler().fit(flat)
        scaled = scaler.transform(flat).reshape(shape)
        return scaled, scaler

    def standardize(self,data, scaler):
        shape = data.shape
        flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
        scaled = scaler.transform(flat).reshape(shape)
        return scaled
    
    def gen_scaler(self,train_X, b_save =False,data_root =None):
        # scaler update
        self.scaler = self.partial_fit(train_X, self.scaler)
        if(b_save):
            # # scaler 저장하기
            joblib.dump(self.scaler,os.path.join(data_root,f'{data_root}/mixamo.pkl'))
            # scaler 불러오기
            self.scaler = joblib.load(os.path.join(data_root,f'{data_root}/mixamo.pkl'))
        return self.scaler
    def gen_scaler_env(self,train_X, b_save =False,data_root =None):
        # scaler update
        # data -> pose velocity 만
        data_copy = train_X.copy()
        pose_X = data_copy[:,:,:(22*3 + 3)]
        ee_X = data_copy[:,:,(22*3 + 3):(22*3 + 3)+5*3*2]
        vel_X = data_copy[:,:,-4:-1]
        train_X = np.concatenate((pose_X,ee_X,vel_X),axis=-1)
        # scaler partial fit
        self.scaler = self.partial_fit(train_X, self.scaler)
        if(b_save):
            # # scaler 저장하기
            joblib.dump(self.scaler,os.path.join(data_root,f'{data_root}/mixamo.pkl'))
            # scaler 불러오기
            self.scaler = joblib.load(os.path.join(data_root,f'{data_root}/mixamo.pkl'))
        return self.scaler
    
    """
    Condition Data
    
    """
    def concat_sequence_3(self,seqlen, data):
        """ 
        Concatenates a sequence of features to one.
        """
        nn,n_timesteps,n_feats = data.shape
        L = n_timesteps-(seqlen-1)
        inds = np.zeros((L, seqlen)).astype(int)

        #create indices for the sequences we want
        rng = np.arange(0, n_timesteps)
        for ii in range(0,seqlen):  
            inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])  

        #slice each sample into L sequences and store as new samples 
        cc=data[:,inds,:].copy()

        #print ("cc: " + str(cc.shape))

        #reshape all timesteps and features into one dimention per sample
        dd = cc.reshape((nn, L, seqlen*n_feats))
        #print ("dd: " + str(dd.shape))
        return dd        
    
    def create_Sequence_CondOnly_OutputData3(self,seqlen,data,num_ee,num_vel,num_env,num_label):
        
        # sequence data joint(198) + vel(3) + env(2830) + label(1)
        joint_data = data[:,:,:-(num_ee + num_vel+num_env+num_label)] # joint data
        joint_num = joint_data.shape[-1]
        if num_env is not 0 :
            ee_data = data[:,:,joint_num:(joint_num+num_ee)] # control data
            control_data = data[:,:,(joint_num+num_ee):(joint_num+num_ee + num_vel) ]
            label_data = data[:,:,-num_label:] # label data
        else:
            control_data = data[:,:,-(num_vel+num_label):] # control data
            label_data = data[:,:,:0] # label data 
        # 
        env_data = data[:,:,(joint_data.shape[-1]+num_ee+num_vel):-(num_label)]
        

        # current pose (output)
        n_frames = joint_data.shape[1]
        new_x = self.concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
        new_env = self.concat_sequence_3(1, env_data[:,seqlen:n_frames,:])
        new_label = self.concat_sequence_3(1,label_data[:,seqlen:n_frames,:])
        new_ee = self.concat_sequence_3(1,ee_data[:,seqlen:n_frames,:])
        # control autoreg(10) + control(11 or 1)
        autoreg_control = self.concat_sequence_3(seqlen +1, control_data)
        single_control = self.concat_sequence_3(1, control_data[:,seqlen:n_frames,:])
        
        #
        autoreg_seq = self.concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
        autoreg_seq_control = np.concatenate((autoreg_seq,autoreg_control),axis=-1)
        autoreg_seq_single_control = np.concatenate((autoreg_seq,single_control),axis=-1)
        
        return new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control, new_ee,new_env,new_label
    
    
    def create_trainable_data(self, seqlen, data, condinfo, b_save=False, data_root=None, b_label=False):
        # scaling
        env_dim = 2640
        data_copy = data.copy()
        if (b_label == True):
            label_id = 1
            pose_X = data_copy[:,:,:(22*3+3)].copy()
            ee_X = data_copy[:,:,(22*3+3):(22*3+3)+5*3*2]
            vel_X = data_copy[:,:,-4:-1]
            env_occupancy_X = data_copy[:,:,-(env_dim+4):-(4)]
            label_Y = data[...,-(label_id):].copy()

        scaled_data_X = self.standardize(np.concatenate((pose_X,ee_X,vel_X),axis=-1),self.scaler)
        
        if(b_label == True):
            scaled_data_X = np.concatenate((scaled_data_X,env_occupancy_X),axis=-1)
            scaled_data_X = np.concatenate((scaled_data_X,label_Y),axis=-1)
        np.savez_compressed(f'{data_root}_all.npz', clips = data_copy)
    
        if(b_save):
            # trainable data
            new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control,new_ee,new_env,new_label = self.create_Sequence_CondOnly_OutputData3(seqlen,scaled_data_X,30,condinfo,env_dim,1)
            #save
            datafilecnt_train =0
            for i in range(0,new_x.shape[0]):
                print("datafilecnt"+str(datafilecnt_train))
                np.savez_compressed(f'{data_root}_scaled_x_{str(datafilecnt_train)}.npz', clips = new_x[i,...])
                np.savez_compressed(f'{data_root}_scaled_env_{str(datafilecnt_train)}.npz', clips = new_env[i,...])
                np.savez_compressed(f'{data_root}_scaled_label_{str(datafilecnt_train)}.npz', clips = new_label[i,...])
                np.savez_compressed(f'{data_root}_scaled_ee_{str(datafilecnt_train)}.npz', clips = new_ee[i,...])
               
                np.savez_compressed(f'{data_root}_scaled_seqControl_{str(datafilecnt_train)}.npz', clips = autoreg_control[i,...])
                np.savez_compressed(f'{data_root}_scaled_singleControl_{str(datafilecnt_train)}.npz', clips = single_control[i,...])
            
                np.savez_compressed(f'{data_root}_scaled_seqControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_control[i,...])
                np.savez_compressed(f'{data_root}_scaled_singleControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_single_control[i,...])
                
                datafilecnt_train += 1
    
    def save_split_test_data(self,data,data_root,b_save=False):
        # scaling
        scaled_data_X = data.copy()
        
        if(b_save):
            datafilecnt = 0
            for i_te in range(0, data.shape[0]):
                print("datafilecnt"+str(datafilecnt))
                np.savez_compressed(f'{data_root}_scaled_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,...])
                datafilecnt += 1    
        return scaled_data_X
                
    def save_split_trainable_data(self,seqlen, data, condinfo, b_save=False, data_root=None):
        # scaling
        scaled_data_X = data.copy()
    
        if(b_save):
            # trainable data
            new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control = self.create_Sequence_CondOnly_OutputData3(seqlen,scaled_data_X,condinfo)
            #save
            datafilecnt_train =0
            for i in range(0,new_x.shape[0]):
                print("datafilecnt_train"+str(datafilecnt_train))
                np.savez_compressed(f'{data_root}_scaled_seqX_{str(datafilecnt_train)}.npz', clips = new_x[i,...])
                
                np.savez_compressed(f'{data_root}_scaled_seqControl_{str(datafilecnt_train)}.npz', clips = autoreg_control[i,...])
                np.savez_compressed(f'{data_root}_scaled_singleControl_{str(datafilecnt_train)}.npz', clips = single_control[i,...])
            
                np.savez_compressed(f'{data_root}_scaled_seqControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_control[i,...])
                np.savez_compressed(f'{data_root}_scaled_singleControlAutoreg_{str(datafilecnt_train)}.npz', clips = autoreg_seq_single_control[i,...])
                
                datafilecnt_train += 1
        return self.create_Sequence_CondOnly_OutputData3(seqlen,scaled_data_X,condinfo)

    def create_test_data(self,data,data_root,b_save=False,b_label = True):
        # scaling
        env_dim = 2640
        data_copy = data.copy()
        if (b_label == True):
            label_id = 1
            pose_X = data_copy[:,:,:22*3+3].copy()
            ee_X = data_copy[:,:,(22*3+3):(22*3+3)+5*3*2]
            vel_X = data_copy[:,:,-4:-1]
            env_occupancy_X = data_copy[:,:,-(env_dim+4):-(4)]
            label_Y = data[...,-(label_id):].copy()

        scaled_data_X = self.standardize(np.concatenate((pose_X,ee_X,vel_X),axis=-1),self.scaler)
        #
        if(b_label == True):
            scaled_data_X = np.concatenate((scaled_data_X,env_occupancy_X),axis=-1)
            scaled_data_X = np.concatenate((scaled_data_X,label_Y),axis=-1)
            
        np.savez_compressed(f'{data_root}_scaled_all.npz', clips = scaled_data_X)

        if(b_save):
            datafilecnt = 0
            for i_te in range(0, data.shape[0]):
                print("datafilecnt"+str(datafilecnt))
                np.savez_compressed(f'{data_root}_scaled_{str(datafilecnt)}.npz', clips = scaled_data_X[i_te,...])
                datafilecnt += 1    
    """
    
    World position data

    """
    def gen_world_pos_Env(self,i_points,i_occ,i_rootpos,i_rootfor):
        joints = i_points.copy()
        occupancy = i_occ.copy()
        root_pos = i_rootpos.copy()
        root_for = i_rootfor.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            
            translation = translation + rot.apply(np.array([root_pos[i,0], 0, root_pos[i,1]])) # next translation
            forward = rot.apply(np.array([root_for[i,0], 0, root_for[i,1]]))
            up = np.array([0, 1, 0])
            xaxis = np.cross(up,forward)
            rot = R.from_matrix([[xaxis[0],  up[0],  forward[0]        ],
                                 [xaxis[1],  up[1],  forward[1]        ],
                                 [xaxis[2],  up[2],  forward[2]        ]]) # next rotation


            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = (joints[i,:,0] + translation[0,0])
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
            

            translations[i,:] = translation

            
        
        return joints, occupancy

    def gen_world_pos(self,i_joints,i_rootpos,i_rootfor):
        joints = i_joints.copy()
        root_pos = i_rootpos.copy()
        root_for = i_rootfor.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            translation = translation + rot.apply(np.array([root_pos[i,0], 0, root_pos[i,1]]))
            forward = rot.apply(np.array([root_for[i,0], 0, root_for[i,1]]))
            up = np.array([0, 1, 0])
            xaxis = np.cross(up,forward)
            rot = R.from_matrix([[xaxis[0],  up[0],  forward[0]        ],
                                 [xaxis[1],  up[1],  forward[1]        ],
                                 [xaxis[2],  up[2],  forward[2]        ]])
            # rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot
            
            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = (joints[i,:,0] + translation[0,0])
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
                       
            
            translations[i,:] = translation
        
        return joints, translations

    def gen_world_pos_data(self,i_joints,i_rootvel):
        joints = i_joints.copy()
        roots = i_rootvel.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))
        
        root_dx, root_dz, root_dr = roots[...,-3], roots[...,-2], roots[...,-1]
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot
            
            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = joints[i,:,0] + translation[0,0]
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
                       
            
            translations[i,:] = translation
        
        return joints, translations

    def gen_world_targetdata(self,i_joints,i_rootvel):
        joints = i_joints.copy()
        roots = i_rootvel.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))

        root_pos = roots[...,:3]
        root_rot_forward = roots[...,3:6]
        root_rot_right = roots[...,6:9]
        
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            translation = translation + rot.apply(np.array([root_pos[i,0], 0, root_pos[i,2]])) # next translation
            forward = rot.apply(np.array([root_rot_forward[i,0], 0, root_rot_forward[i,2]]))
            xaxis = rot.apply(np.array([root_rot_right[i,0], 0, root_rot_right[i,2]]))
            
            rot = R.from_matrix([[xaxis[0],  0,  forward[0]        ],
                                 [xaxis[1],  1,  forward[1]        ],
                                 [xaxis[2],  0,  forward[2]        ]]) # next rotation


            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = (joints[i,:,0] + translation[0,0])
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
            

            translations[i,:] = translation
        
        return joints, translations


    def gen_world_Target(self,i_toTarget,i_rootvel):
        target = i_toTarget.copy()
        roots = i_rootvel.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((target.shape[0],3))
        
        root_dx, root_dz, root_dr = roots[...,-3], roots[...,-2], roots[...,-1]
        
        for i in range(len(target)):
            
            targetposition = rot.apply(target[i]) # velocity
            translations[i,:] = targetposition + translation

            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot
            
        return translations

    def gen_world_pos_Envdata(self,i_points,i_occ,i_rootvel):
        joints = i_points.copy()
        occupancy = i_occ.copy()
        roots = i_rootvel.copy()

        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((joints.shape[0],3))
        
        root_dx, root_dz, root_dr = roots[:,-3], roots[:,-2], roots[:,-1]
        joints = joints.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
            
            joints[i,:,:] = rot.apply(joints[i])
            joints[i,:,0] = joints[i,:,0] + translation[0,0]
            joints[i,:,2] = joints[i,:,2] + translation[0,2]
            
            
            translations[i,:] = translation
        
        return joints, occupancy
    # def gen_world_pos_data(self,input_data):
    #     un_scaled_data = input_data.copy()
    #     rot = R.from_quat([0,0,0,1])
    #     translation = np.array([[0,0,0]])
    #     translations = np.zeros((un_scaled_data.shape[0],3))
        
    #     joints, root_dx, root_dz, root_dr = un_scaled_data[:,:66], un_scaled_data[:,-3], un_scaled_data[:,-2], un_scaled_data[:,-1]
        
    #     joints = joints.reshape((len(joints), -1, 3))
    #     for i in range(len(joints)):
            
    #         joints[i,:,:] = rot.apply(joints[i])
    #         joints[i,:,0] = joints[i,:,0] + translation[0,0]
    #         joints[i,:,2] = joints[i,:,2] + translation[0,2]
            
    #         rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot
    #         #translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
    #         translation = translation + np.array([root_dx[i], 0, root_dz[i]])
            
    #         translations[i,:] = translation
        
    #     return joints, translations

    def gen_world_pos_data_differentFrame(self,input_data):
        un_scaled_data = input_data.copy()
        rot = R.from_quat([0,0,0,1])
        translation = np.array([[0,0,0]])
        translations = np.zeros((un_scaled_data.shape[0],3))
        
        joints, root_dx, root_dz, root_dr = un_scaled_data[:,:66], un_scaled_data[:,-3], un_scaled_data[:,-2], un_scaled_data[:,-1]
        joints_ee = un_scaled_data[:,66:66+15]
        joints = joints.reshape((len(joints), -1, 3))
        joints_ee = joints_ee.reshape((len(joints), -1, 3))
        for i in range(len(joints)):
            
            # (0) : Hip (seen by root)
            joints[i,0,:] = rot.apply(joints[i,0])
            joints[i,0,0] = joints[i,0,0] + translation[0,0]
            joints[i,0,2] = joints[i,0,2] + translation[0,2]
            
            # (14~22) : Lower body (seen by root)
            joints[i,14:,:] = rot.apply(joints[i,14:])
            joints[i,14:,0] = joints[i,14:,0] + translation[0,0]
            joints[i,14:,2] = joints[i,14:,2] + translation[0,2]
            
            # (1~13) : Upper Body (seen by hip)
            joints[i,1:14,:] = rot.apply(joints[i,1:14])
            joints[i,1:14,0] = joints[i,1:14,0] + joints[i,0,0]
            joints[i,1:14,1] = joints[i,1:14,1] + joints[i,0,1]
            joints[i,1:14,2] = joints[i,1:14,2] + joints[i,0,2]

            #
            # (14~22) : EE body (seen by root)
            joints_ee[i,:,:] = rot.apply(joints_ee[i])
            joints_ee[i,:,0] = joints_ee[i,:,0] + translation[0,0]
            joints_ee[i,:,2] = joints_ee[i,:,2] + translation[0,2]

            rot = R.from_rotvec(np.array([0,-root_dr[i],0]))*rot
            translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
            #translation = translation + np.array([root_dx[i], 0, root_dz[i]])
            
            translations[i,:] = translation
        
        return joints, translations, joints_ee

    """
    
    data augmentation

    """
    def mirror_data(self,data):
        aa = data.copy()
        aa[:,:,3*18:3*(21+1)]=data[:,:,3*14:3*(17+1)] # leftupleg, rightupleg
        aa[:,:,3*18:3*(21+1):3]=-data[:,:,3*14:3*(17+1):3] # leftupleg, rightupleg
        aa[:,:,3*14:3*(17+1)]=data[:,:,3*18:3*(21+1)] # rightupleg, leftupleg
        aa[:,:,3*14:3*(17+1):3]=-data[:,:,3*18:3*(21+1):3] # rightupleg, leftupleg
        
        aa[:,:,3*6:3*(9+1)]=data[:,:,3*10:3*(13+1)] # leftshoulder,rightshoulder
        aa[:,:,3*6:3*(9+1):3]=-data[:,:,3*10:3*(13+1):3] #
        aa[:,:,3*(10):3*(13+1)]=data[:,:,3*6:3*(9+1)] # rightshoulder
        aa[:,:,3*(10):3*(13+1):3]=-data[:,:,3*6:3*(9+1):3]
        aa[:,:,66]=-data[:,:,66]
        aa[:,:,68]=-data[:,:,68]
        
        return aa

    def reverse_time(self,data):
        aa = data[:,-1::-1,:].copy()
        aa[:,:,66] = -aa[:,:,66]
        aa[:,:,67] = -aa[:,:,67]
        aa[:,:,68] = -aa[:,:,68]
        return aa

    def gen_augmentation(self,train_X):
        train_X = self.mirror_data(train_X)
        return train_X
    
    """
    data generation procedure

    """
    def gen_learning_data_fromTXT(self,data_root, bool_save= False, window = 80,window_step=81, num_files =10000, Train=False, target_isMat = False):
        """ load txt files & create output data folder"""
        cmu_files = self.get_txtfiles(f'{data_root}')
        learning_datas = []
        draw_datas = []
        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz_position'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)
        
        """ generate learning data from raw txt file """
        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            if(target_isMat):
                learningdata, drawdata = self.process_file_2023_txt_position(item,window=window,window_step=window_step,Train=Train) # draw_data
            else:
                learningdata, drawdata = self.process_file_2023PG_txt_position(item,window=window,window_step=window_step,Train=Train) # draw_data
                
            learning_datas+=(learningdata)
            #draw_datas+=(drawdata)
            if (i == num_files):
                break;
            
        data_clips = np.array(learning_datas)
        data_clips_draw = np.array(draw_datas)
        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            #np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced_draw', clips=data_clips_draw)
                    
        return data_clips,data_clips_draw


    def gen_npz_data_fromTXT_inFolder_position(self,data_root, bool_save= False, FPS_60_divide=1, window = 80, filterwidth=20, num_files =10000, label=None,b_test=False):
        
        cmu_files = self.get_txtfiles(f'{data_root}')
        cmu_clips = []
        draw_clips = []
        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz_position'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)

        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            # normal
            clips,drawdata,_,filtered_drawdata = self.process_file_0530_txt_position(item,divide=FPS_60_divide,window=window,filterwidth=filterwidth,label=label) # draw_data
            #np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',drawdata,delimiter=" ")
            cmu_clips += clips
            draw_clips += filtered_drawdata
            if (i == num_files):
                break;
            if(b_test):
                np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',clips[0],delimiter=" ")
        
        data_clips = np.array(cmu_clips)
        data_clips_draw = np.array(draw_clips)
        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced_draw', clips=data_clips_draw)
                    
        return data_clips,data_clips_draw

    def gen_npz_data_fromTXT_inFolder_position_rotation(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =10000, label=None,b_test=False):
        
        cmu_files = self.get_txtfiles(f'{data_root}')
        cmu_clips = []

        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz_position_rotation'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)

        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            # normal
            clips,draw_data, _ = self.process_file_0530_txt_position_rotation(item,divide=FPS_60_divide,filterwidth=filterwidth,label=label)
            cmu_clips += clips
            if (i == num_files):
                break;
            if(b_test):
                np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',clips[0],delimiter=" ")
        
        data_clips = np.array(cmu_clips)

        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            
                    
        return data_clips

    def gen_npz_data_fromTXT_inFolder_hip_ori(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =10000, label=None,b_test=False):
        
        cmu_files = self.get_txtfiles(f'{data_root}')
        cmu_clips = []

        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz_hip_ori'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)

        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            # normal
            clips,draw_data, _ = self.process_file_0530_txt_hip_ori(item,divide=FPS_60_divide,filterwidth=filterwidth,label=label)
            cmu_clips += clips
            if (i == num_files):
                break;
            if(b_test):
                np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',clips[0],delimiter=" ")
        
        data_clips = np.array(cmu_clips)
       
        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            
                    
        return data_clips
    """
    SAMP example
    """
    def gen_npz_data_fromTXT_inFolder_SAMP_position(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =10000, label=None,b_test=False):
        
        cmu_files = self.get_txtfiles(f'{data_root}')
        cmu_clips = []

        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz_position'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)
    
        print('Processing %i of %i (%s)' % (0, len(cmu_files), cmu_files[0]))
        print('Processing %i of %i (%s)' % (1, len(cmu_files), cmu_files[1]))
        
        # normal
        clips,drawdata,_,_ = self.process_file_0530_txt_SAMP_position(cmu_files[0],cmu_files[1],divide=FPS_60_divide,filterwidth=filterwidth,label=label) # draw_data
        #np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',drawdata,delimiter=" ")
        cmu_clips += clips
        
        if(b_test):
            np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(cmu_files[0])}_draw.txt',clips[0],delimiter=" ")
        
        data_clips = np.array(cmu_clips)

        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            
                    
        return data_clips

    def gen_npz_data_fromTXT_inFolder(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =10000, label=None,b_test=False):
        
        cmu_files = self.get_txtfiles(f'{data_root}')
        cmu_clips = []

        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)

        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            # normal
            clips,draw_data, _ = self.process_file_0530_txt(item,divide=FPS_60_divide,filterwidth=filterwidth,label=label)
            cmu_clips += clips
            if (i == num_files):
                break;
            if(b_test):
                np.savetxt(f'{create_dir_save_trainable_data}/{os.path.basename(item)}_draw.txt',clips[0],delimiter=" ")
        
        data_clips = np.array(cmu_clips)

        if(bool_save):
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            
                    
        return data_clips

    def gen_npz_data_fromBVH_inFolder(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =1000, label=None):
        
        cmu_files = self.get_bvhfiles(f'{data_root}')
        cmu_clips = []
        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            clips, _ = self.process_file(item,divide=FPS_60_divide,filterwidth=filterwidth)
            cmu_clips += clips
            if (i == num_files):
                break;
            
        data_clips = np.array(cmu_clips)
        
        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)
            np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
                    
        return data_clips

    def gen_learnable_data_inFolder(self,data_root, b_split_data):
        
        bvh_files = self.get_npzfiles(data_root)

        data_X = np.load(bvh_files[0])['clips'].astype(np.float32)
        for i in range(1,len(bvh_files)):
            print ('processing %i of %i (%s)' % (i, len(bvh_files),bvh_files[i]))
            bvh_file = bvh_files[i]
            # load clip
            clip_data_X = np.load(bvh_file)['clips'].astype(np.float32)
            
            #condatenation
            data_X = np.concatenate((data_X, clip_data_X),axis = 0)
        
        if (b_split_data == False):
            # all data is a train data
            train_X = data_X
            return train_X, np.zeros(1), np.zeros(1)
        else:
            # data_X -> train, valid , test
            train_X, valid_X = train_test_split(data_X, test_size=0.2,random_state=1004) # train valid 나눠서 sequence data로 만듬 
            valid_X, test_X = train_test_split(valid_X, test_size=0.1,random_state=1004) # train valid 나눠서 sequence data로 만듬 
            
            train_X = np.concatenate((train_X,valid_X[100:,...]),axis=0)
            train_X = np.concatenate((train_X,test_X[30:,...]), axis=0)
            valid_X = valid_X[:100,...]
            test_X = test_X[:30,...]
            
        return train_X, valid_X, test_X
    """
    bvh files in folder -> train, valid, test 
    """
    def gen_npz_from_BVH_inFolder(self,data_root, bool_save= False, FPS_60_divide=1, filterwidth=20, num_files =1000, label=None):
        # bvh files to sliced npz
        cmu_files = self.get_bvhfiles(f'{data_root}')
        cmu_clips = []
        for i, item in enumerate(cmu_files):
            
            print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
            clips, _ = self.process_file(item,divide=FPS_60_divide,filterwidth=filterwidth,label=label)
            cmu_clips += clips
            if (i == num_files):
                break;
            
        data_clips = np.array(cmu_clips)
        
        #augmentation
        train_X_copy = data_clips.copy()
        if (label is not None ):
            label_id = 1
            train_X_copy = data_clips[...,:-(label_id)].copy()
                    
        mirror_X = self.mirror_data(train_X_copy)
        reverse_X = self.reverse_time(train_X_copy)
        train_X_copy = np.concatenate((train_X_copy,mirror_X),axis=0)
        train_X_copy = np.concatenate((train_X_copy,reverse_X),axis=0)

        if (label is not None):
            label_X_copy = data_clips[...,-(label_id):].copy()
            label_X = data_clips[...,-(label_id):].copy()
            label_X_copy = np.concatenate((label_X_copy,label_X),axis=0)
            label_X_copy = np.concatenate((label_X_copy,label_X),axis=0)
            train_X_copy = np.concatenate((train_X_copy,label_X_copy),axis=-1)

        if (bool_save):
            basename = os.path.basename(data_root)
            create_dir_save_trainable_data = f'{data_root}/npz'
            if not os.path.exists(create_dir_save_trainable_data):
                os.makedirs(create_dir_save_trainable_data)
            if label is not None :
                np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced_withlabel', clips=train_X_copy)
            else :
                np.savez_compressed(f'{create_dir_save_trainable_data}/{basename}_sliced', clips=data_clips)
            
                    
        return train_X_copy
    
    def gen_Train_Valid_Test_Data(self,data_root,b_split_data=False,b_label=False, b_first=False, b_save_set = False):
        #npz file -> sequence split
        train_X, valid_X, test_X = self.gen_learnable_data_inFolder(f'{data_root}',b_split_data)
        
        #training_data
        create_dir_save_trainable_data = f'{data_root}/npz'
        if not os.path.exists(create_dir_save_trainable_data):
            os.makedirs(create_dir_save_trainable_data)
        
        # scaler 만들기
        if b_first == True :
            self.scaler = self.gen_scaler_env(train_X,True,f'{create_dir_save_trainable_data}')
            self.create_trainable_data(10,train_X,3,b_save_set,f"{create_dir_save_trainable_data}/train",b_label = True)
        if b_first == False :
            self.scaler = joblib.load(os.path.join(data_root,f'{create_dir_save_trainable_data}/mixamo.pkl'))
            self.scaler = self.gen_scaler_env(train_X,True,f'{create_dir_save_trainable_data}')
            self.create_trainable_data(10,train_X,3,b_save_set,f"{create_dir_save_trainable_data}/train",b_label = True)
        #scaler
        if(b_label==True and b_split_data==True): 
            label_id = 1
            self.create_trainable_data(10,train_X,3,b_save_set,f"{create_dir_save_trainable_data}/train",b_label = True)
            self.create_trainable_data(10,valid_X,3,b_save_set,f"{create_dir_save_trainable_data}/valid",b_label = True)
            self.create_test_data(test_X,f"{create_dir_save_trainable_data}/test",b_save_set,b_label = True)
        elif(b_label==False and b_split_data==True) :
            self.create_trainable_data(10,train_X,3,b_save_set,f"{create_dir_save_trainable_data}/train")
            self.create_trainable_data(10,valid_X,3,b_save_set,f"{create_dir_save_trainable_data}/valid")
            self.create_test_data(test_X,f"{create_dir_save_trainable_data}/test",b_save_set,b_label = False)
           
    #------------------setup   
    def save(self,save_root,data_X):
        np.savez_compressed(save_root, clips = data_X)

    
    #-----------update
    def np3_to_Vector3(self,np3):
        ry_vec = pr.Vector3(0,0,0)
        ry_vec.x = np3[0]
        ry_vec.y = np3[1]
        ry_vec.z = np3[2]
        return ry_vec

    def update_Scale_data(self, data, scaler=None):
        if(len(data.shape)) == 2:
            scaled_data = scaler.transform(data)
        else:
            scaled_data = self.standardize(data,scaler)
        return scaled_data
        
    def update_UnScale_data(self, data, scaler=None):
        scaler = self.scaler
        if(len(data.shape)) == 2:
            un_scaled_data = scaler.inverse_transform(data)
        else:
            un_scaled_data = self.inv_standardize(data,scaler)
        return un_scaled_data

    def update_batch(self,nBatch):
        self.batch_X = self.loadMotionData(nBatch)
        self.update_UnScale_data(self.batch_X,self.scaler)

    def update_drawEnvs(self,nGlobalCnt,input_pnts,occupancies,color):
        env_points = input_pnts.copy()
        env_occupancy = occupancies.copy()
        for j in range(env_occupancy.shape[1]):
            if env_occupancy[nGlobalCnt,j] > 0.9 :
                DrawSphere(self.np3_to_Vector3(env_points[nGlobalCnt,j,:]),0.03,color)

    def update_drawJointsEE(self,nGlobalCnt, input_joints,input_joints_ee, color):
        joints = input_joints.copy()
        joints_ee = input_joints_ee.copy()
        # draw joints
        for j in range(len(self.parents_ee)):
            if self.parents_ee[j] != -1:
                DrawLine3D(self.np3_to_Vector3(joints_ee[nGlobalCnt,j,:]),self.np3_to_Vector3(joints[nGlobalCnt,self.parents_ee[j],:]),color)

    def update_drawJoints(self,nGlobalCnt, input_joints, color):
        joints = input_joints.copy()
        # draw joints
        for j in range(len(self.parents)):
            if self.parents[j] != -1:
                DrawLine3D(self.np3_to_Vector3(joints[nGlobalCnt,j,:]),self.np3_to_Vector3(joints[nGlobalCnt,self.parents[j],:]),color)
    
    def update_drawTrajectory(self,translations,color):
        for i in range(translations.shape[0]-1):
            DrawLine3D(self.np3_to_Vector3(translations[i,:]),self.np3_to_Vector3(translations[i+1,:]),color)

    def update_drawToTarget(self,translations,target,color):
        for i in range(translations.shape[0]-1):
            DrawLine3D(self.np3_to_Vector3(translations[i,:]),self.np3_to_Vector3(target[i,:]),color)

       
        