import argparse

import numpy as np
import torch
import torch.nn as nn
from diff_models_csdi import diff_CSDI


from dataset_netdata import Area_nums
import matplotlib.pyplot as plt
# def split_time_frequency(x):
    #
    #
    # q = 5
    # x_ft = torch.fft.rfft(x, axis=-1)
    # B, K, L = x_ft.shape
    # # 将张量变形为 (B * K, L) 的二维张量
    # reshaped_tensor = x_ft.view(B * K, L)
    #
    # # 找到每行中最大的 q 个值的索引
    # _, top_indices = torch.topk(abs(reshaped_tensor), q, dim=1)
    #
    # # 创建一个与原张量形状相同的零张量
    # result = torch.zeros_like(x_ft)
    #
    # # 将最大的 q 个值赋值给结果张量
    # result.view(B * K, L)[torch.arange(B * K,).unsqueeze(1), top_indices] = reshaped_tensor[
    #     torch.arange(B * K,).unsqueeze(1), top_indices]
    #
    # # 将结果张量还原为原形状
    # result = result.view(B, K, L)
    # ifft_x = torch.fft.irfft(result)
    # residual_x = (x - ifft_x).double()
    # return ifft_x,residual_x



class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.mse = nn.MSELoss()
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = torch.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = torch.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)


    def calc_loss_valid(
        self, observed_data, is_train,idex_test
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, is_train, idex_test, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps
    def noise_images(self, x, t):
        t = t.to(self.alpha_hat.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        Ɛ = torch.randn_like(x).to(self.alpha_hat.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    def calc_loss(
        self, observed_data,  is_train, idex_test,set_t=-1
    ):
        B, K, L = observed_data.shape

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            #current_alpha = self.alpha_torch[t]  # (B,1,1)
           # observed_data = torch.fft.rfft(observed_data, axis=-1)
           # noise = torch.randn_like(observed_data)
            #noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise #noisy+data
            noisy_data, noise = self.noise_images(observed_data,t)

            total_input = self.set_input_to_diffmodel(noisy_data, observed_data)#here
  
            predicted = self.diffmodel(total_input, t,idex_test)  # (B,K,L)
           # loss =self.mse(noise,predicted.permute(2, 0, 1))
            loss  = self.mse(noise,predicted.permute(0,2,1)).to(self.alpha_hat.device)
        return loss
 

    def set_input_to_diffmodel(self, noisy_data, observed_data):
        if self.is_unconditional == True:
            # total_input = noisy_data.unsqueeze(-1)  # (B,K,L,1)
            total_input = noisy_data  # (B,K,L,1)
            # total_input =torch.cat([noisy_data, observed_data], dim=1) # (B,2,K,L)
        else:
            cond_obs = observed_data.unsqueeze(1)
            noisy_target = noisy_data.unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data,idex_test,n_samples):
        # observed_data = observed_data[:,:,:48]
        B, K, L = observed_data.shape

        # _, _, E = observed_fft.shape

        # observed_data = observed_data.unsqueeze(1)



        imputed_samples = torch.zeros( B, n_samples,K, L).to(self.device)
        for i in range(n_samples):
            x = torch.randn_like(observed_data)


            # current_sample = torch.stack((in_one_one, in_two_two,), dim=1)  # B,C=2,K,E
            for t in reversed(range(1, self.num_steps)):
                diff_input = x


                predicted = self.diffmodel(diff_input, torch.tensor([t]).to(self.device),idex_test)
                alpha = self.alpha[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                alpha_hat = self.alpha_hat[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                beta = self.beta[t].unsqueeze(-1).unsqueeze(-1).to(self.device)
                # coeff1 = 1 / self.alpha_hat[t] ** 0.5
                # coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                # current_sample = coeff1 * (x - coeff2 * predicted)

                if t > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # aaaa=torch.sqrt(alpha)
                # bbbb=torch.sqrt(1 - alpha_hat)
                # cccc=1 - alpha
                # dddd = cccc/bbbb
                # eeee = dddd* predicted
                # ffff = x -eeee
                # gggg = torch.sqrt(beta) * noise

                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted.permute(0, 2, 1)) + torch.sqrt(
                    beta) * noise


            record_sample = x
                # if t > 0:


            imputed_samples[:,i] = record_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_tp,
            idex_test,
        ) = self.process_data(batch)


        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, is_train,idex_test)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_tp ,
            idex_test,
        ) = self.process_data(batch)


        with torch.no_grad():
            # cond_mask = gt_mask
            # target_mask = observed_mask - cond_mask

            # side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data,idex_test, n_samples)

            # for i in range(len(cut_length)):  # to avoid double evaluation
            #     target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data,observed_tp


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=Area_nums()):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        idex_test = batch["idex_test"].to(self.device).int()
        #side_info = batch['side_info'].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)

        # cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        # for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_tp,
            idex_test,
        )
