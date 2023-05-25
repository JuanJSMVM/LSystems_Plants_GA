import plot_results
import numpy as np

folders=["SelRuleta_MutSwap","SelRuleta_MutFlip",
         "SelTorneo_MutFlip","SelTorneo_MutSwap",
         "SelRank_MutFlip","SelRank_MutSwap"]
elitism=["SinElite_","Elite_"]
size_ax=[12,16]

def create_graphs_exps(best_ind_arr,best_gen_inds_arr,size):
    fold_num=0
    for idx in range(best_ind_arr.shape[0]):
        folder=""
        print(idx,fold_num)
        if idx <=5:
            folder+=elitism[0]
            folder+=folders[fold_num]
            folder+="_"+str(size)
        else:
            if idx==6:
                fold_num=0
            folder+=elitism[1]
            

            folder+=folders[fold_num]
            folder+="_"+str(size)
        best_ind=best_ind_arr[idx]

        best_gen_inds=best_gen_inds_arr[idx].reshape(700,size)
        best_gen_inds=best_gen_inds[::10]

        plot_results.create_env(best_ind,best_gen_inds,folder)
        fold_num+=1

def create_animations(shape,size):
    fold_num=0
    for idx in range(shape):
        folder=""
        
        if idx <=5:
            folder+=elitism[0]
            folder+=folders[fold_num]
            folder+="_"+str(size)
        else:
            if idx==6:
                fold_num=0
            folder+=elitism[1]
            

            folder+=folders[fold_num]
            folder+="_"+str(size)
        #print(folder)    
        plot_results.create_gif(folder)
        fold_num+=1




best_final_exp1=np.load('best_final_ind_exp1.npy',allow_pickle=True) # load
best_final_exp2=np.load('best_final_ind_exp2.npy',allow_pickle=True) # load
best_gen_inds_exp1 = np.load('best_gen_inds_exp1.npy',allow_pickle=True) # load
best_gen_inds_exp2 = np.load('best_gen_inds_exp2.npy',allow_pickle=True) # load

create_animations(best_final_exp1.shape[0],12)
create_animations(best_final_exp2.shape[0],16)
#create_animations(best_final_exp2.shape[0],16)
#create_graphs_exps(best_final_exp1,best_gen_inds_exp1,size_ax[0])
#create_graphs_exps(best_final_exp2,best_gen_inds_exp2,size_ax[1])

