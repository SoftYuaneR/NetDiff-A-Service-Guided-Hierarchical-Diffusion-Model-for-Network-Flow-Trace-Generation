import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
from scipy.spatial import distance
#from pytorch_msssim import ssim

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        print("training process")
        print('current epoch:',epoch_no)
        avg_loss = 0
        model.train()
        print(train_loader)
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target.permute(0, 2, 1)) * eval_points * ((target.permute(0, 2, 1) <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def crps(y_true, y_pred, sample_weight=None):
     num_samples=y_pred.shape[0]
     absolute_error=np.mean(np.abs(y_pred-y_true), axis=0)
 
     if num_samples==1:
         return np.average(absolute_error, weights=sample_weight)
 
     y_pred=np.sort(y_pred, axis=0)
     diff=y_pred[1:] -y_pred[:-1]
     weight=np.arange(1, num_samples) *np.arange(num_samples-1, 0, -1)
     weight=np.expand_dims(weight, -1)
 
     per_obs_crps=absolute_error-np.sum(diff*weight, axis=0) /num_samples**2
     return np.average(per_obs_crps, weights=sample_weight)

def distribution_jsd(generated_data, real_dataset):

    
    
    
    n_real = real_dataset.flatten()
    n_gene = generated_data.flatten()
    JSD = distance.jensenshannon(n_real.flatten(), n_gene.flatten(), 2.0)
    
    return JSD
    

    
    
def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    print('testnsample=:',nsample)
    with torch.no_grad():
        model.eval()
        evalpoints_total = 0
        evalpoints_ssim = 0
        js_total = 0
        js_one_total = 0
        eval_js_total = 0
        evalpoints_one_total = 0
        ssim_value = 0
        tv_distance_total0 =0
        tv_distance_total1 =0
        tv_distance_total2 =0
        tv_distance_total3 =0
        all_target = []
        all_observed_time = []
        all_generated_samples = []
        cut=5
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                sample, c_targets, observed_time = output
                samples = sample.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_targets.permute(0, 2, 1)  # (B,L,K)
                
                
                #samples = sample[:,:,cut:-cut,:]
                #c_target = c_targets[:,cut:-cut,:]
                
                #print(c_target.shape)
                B, L, K = c_target.shape

                samples_median = samples.median(dim=1)
                all_target.append(c_target)

                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                evalpoints_total += (B*K*L)


                print(samples.shape)
                print(c_target.shape)
                #print(all_generated_samples.shape)
                #print(all_target.shape)
#----------------------------generated Metric------------------------------------
                # #：Metrix 1----JS Divers

                epsilon = 100
                # js_distance = distance.jensenshannon(samples_median.values.cpu().numpy().reshape(B,K*L).T + epsilon, c_target.cpu().numpy().reshape(B,K*L).T + epsilon, 2.0)
                flatten0_samp =samples[:,:,:,0].cpu().numpy().flatten()
                flatten0_targ = c_target[:,:,0].cpu().numpy().flatten()
                
               # flatten1_samp =samples[:,:,:,1].cpu().numpy().flatten()
               # flatten1_targ = c_target[:,:,1].cpu().numpy().flatten()
               # flatten2_samp =samples[:,:,:,2].cpu().numpy().flatten()
               # flatten2_targ = c_target[:,:,2].cpu().numpy().flatten()
              #  flatten3_samp =samples[:,:,:,3].cpu().numpy().flatten()
              #  flatten3_targ = c_target[:,:,3].cpu().numpy().flatten()


                jsd0 = distribution_jsd((flatten0_samp),(flatten0_targ))
              #  jsd1 = distribution_jsd((flatten1_samp),(flatten1_targ))
              #  jsd2 = distribution_jsd((flatten2_samp),(flatten2_targ))
              #  jsd3 = distribution_jsd((flatten3_samp),(flatten3_targ))

               # aaa0 = (flatten0_samp-flatten0_samp.min())/(flatten0_samp.max()-flatten0_samp.min())
               # norm0_samp = aaa0/aaa0.sum()
               # bbb0 = (flatten0_targ-flatten0_targ.min())/(flatten0_targ.max()-flatten0_targ.min())
                #norm0_targ = bbb0/bbb0.sum()
               # jsd0 = distance.jensenshannon(norm0_samp.reshape(-1,33), norm0_targ.reshape(-1,33), 2.0)
                
                
              #  aaa1 = (flatten1_samp-flatten1_samp.min())/(flatten1_samp.max()-flatten1_samp.min())
              #  norm1_samp = aaa1/aaa1.sum()
                #bbb1 = (flatten1_targ-flatten1_targ.min())/(flatten1_targ.max()-flatten1_targ.min())
              #  norm1_targ = bbb1/bbb1.sum()
             #   jsd1 = distance.jensenshannon(norm1_samp.reshape(-1,33), norm1_targ.reshape(-1,33), 2.0)
                
                
                
              #  aaa2 = (flatten2_samp-flatten2_samp.min())/(flatten2_samp.max()-flatten2_samp.min())
              #  norm2_samp = aaa2/aaa2.sum()
              #  bbb2 = (flatten2_targ-flatten2_targ.min())/(flatten2_targ.max()-flatten2_targ.min())
              #  norm2_targ = bbb2/bbb2.sum()
              #  jsd2 = distance.jensenshannon(norm2_samp.reshape(-1,33), norm2_targ.reshape(-1,33), 2.0)
                
              #  aaa3 = (flatten3_samp-flatten3_samp.min())/(flatten3_samp.max()-flatten3_samp.min())
              #  norm3_samp = aaa3/aaa3.sum()
              #  bbb3 = (flatten3_targ-flatten3_targ.min())/(flatten3_targ.max()-flatten3_targ.min())
              #  norm3_targ = bbb3/bbb3.sum()
              #  
                # #：Metrix 2----1-阶 JS Divers
                

                # #：Metrix 3----TV-Distance
                # #：Metrix 3----TV-Distance
                tv0 = 0
               # tv1 = 0
               # tv2 = 0
               # tv3 = 0
                tf0 = (flatten0_samp.reshape(-1,168))
                tc0 = (flatten0_targ.reshape(-1,168))
                #tf1 = (flatten1_samp.reshape(-1,33))
                #tc1 = (flatten1_targ.reshape(-1,33))
               # tf2 = (flatten2_samp.reshape(-1,33))
                #tc2 = (flatten2_targ.reshape(-1,33))
                #tf3 = (flatten3_samp.reshape(-1,33))
                #tc3 = (flatten3_targ.reshape(-1,33))

                for i in range(len(tc0)):
                    tv0 += 0.5 * abs(tf0[i] - tc0[i])
                tv_distance_res0 = tv0/(168*8457)
                tv_distance_total0 +=tv_distance_res0.sum().item()

               # for i in range(len(tc1)):
               #     tv1 += 0.5 * abs(tf1[i] - tc1[i])
               # tv_distance_res1 = tv1/(33*1672)
               # tv_distance_total1 +=tv_distance_res1.sum().item()
               # 
               # for i in range(len(tc2)):
               #     tv2 += 0.5 * abs(tf2[i] - tc2[i])
               # tv_distance_res2 = tv2/(33*1672)
               # tv_distance_total2 +=tv_distance_res2.sum().item()

              #  for i in range(len(tc3)):
              #      tv3 += 0.5 * abs(tf3[i] - tc3[i])
              #  tv_distance_res3 = tv3/(33*1672)
               # tv_distance_total3 +=tv_distance_res3.sum().item()


                ### #：Metrix 6----SSIM

#                width_base = 8
#                height = B // width_base
#                # target的图像化
##                norm_data_t = (c_target - c_target.min(dim=0,keepdim=True).values) / (c_target.max(dim=0,keepdim=True).values - c_target.min(dim=0,keepdim=True).values)
#                norm_data_t = (c_target - c_target.min()) / (c_target.max() - c_target.min())
##                scaled_data_t = (norm_data_t * 255).to(torch.uint8)
#                image_data_t = norm_data_t.permute(1,2,0).reshape(L, 1, height, width_base).float()
#
#                # generate的图像化
##                norm_data = (samples_median.values - samples_median.values.min(dim=0,keepdim=True).values) / (samples_median.values.max(dim=0,keepdim=True).values - samples_median.values.min(dim=0,keepdim=True).values)
#                norm_data = (samples_median.values - samples_median.values.min()) / (samples_median.values.max() - samples_median.values.min())
##                scaled_data = (norm_data * 255).to(torch.uint8)
#                image_data = norm_data.permute(1,2,0).reshape(L, 1, height, width_base).float()
#                
#                ssim_value_mp = ssim(image_data, image_data_t,data_range=1.0, win_size=3, size_average=False, nonnegative_ssim=True)
#                ssim_value += ssim_value_mp.max().item()
#                evalpoints_ssim += 1
                ssim_value = 1
                evalpoints_ssim = 1
# ----------------------------generated Metric------------------------------------



                it.set_postfix(
                    ordered_dict={
                        "jsd0": jsd0,
                        #"jsd1": jsd1,
                        #"jsd2": jsd2,
                        #"jsd3": jsd3,
                        
                        "tv0":tv_distance_total0,
                        #"tv1":tv_distance_total1,
                        #"tv2":tv_distance_total2,
                       # "tv3":tv_distance_total3,
                        
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS0 = crps(
                (flatten0_samp),(flatten0_targ)
            )
           # CRPS1 = crps(
           #     (flatten1_samp),(flatten1_targ)
           # )
           # CRPS2 = crps(
           #     (flatten2_samp),(flatten2_targ)
           # )
           # CRPS3 = crps(
           #     (flatten3_samp),(flatten3_targ)
           # )


            with open(
                foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        jsd0,
                        #jsd1,
                        #jsd2,
                        #jsd3,
                        tv_distance_total0,
                       # tv_distance_total1,
                       # tv_distance_total2,
                       # tv_distance_total3,
                        CRPS0,
                       # CRPS1,
                       # CRPS2,
                       # CRPS3,
            
                        ssim_value / evalpoints_ssim,
                    ],
                    f,
                )
                
                
                print("the jsd of feature 0:", jsd0)
               # print("the jsd of feature 1:", jsd1)
               # print("the jsd of feature 2:", jsd2)
               # print("the jsd of feature 3:", jsd3)
            
                print("the tv-distance of feature 0:", tv_distance_total0 )
                #print("the tv-distance of feature 1:", tv_distance_total1 )
                #print("the tv-distance of feature 2:", tv_distance_total2 )
                #print("the tv-distance of feature 3:", tv_distance_total3)
                print("CRPS0:", CRPS0)
               # print("CRPS1:", CRPS1)
               # print("CRPS2:", CRPS2)
               # print("CRPS3:", CRPS3)
                print("SSIM:", ssim_value / evalpoints_ssim)
