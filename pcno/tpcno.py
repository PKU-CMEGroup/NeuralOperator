import math
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
from utility.adam import Adam
from utility.losses import LpLoss
from utility.normalizer import UnitGaussianNormalizer
from pcno.geo_utility import compute_edge_gradient_weights
from pcno.pcno import _get_act, compute_Fourier_bases, SpectralConv, compute_gradient





class TPCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        train_sp_L = 'independently',
        act="gelu",
    ):
        super(TPCNO, self).__init__()

        """
            Compared to sdandard PCNO, we have incorporated temporal variables into the framework.
            
            Returns:
                Time-dependent Point cloud neural operator

        """
        self.modes = modes
        self.nmeasures = nmeasures
        

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(
                    self.layers, self.layers[1:]
                )
            ]
        )
        self.train_sp_L = train_sp_L
        self.sp_L = nn.Parameter(torch.ones(ndims, nmeasures), requires_grad = bool(train_sp_L))

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims*in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign

        self.normal_params = []  #  group of params which will be trained normally
        self.sp_L_params = []    #  group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L :
                self.normal_params.append(param)
            else:
                if self.train_sp_L == 'together':
                    self.normal_params.append(param)
                elif self.train_sp_L == 'independently':
                    self.sp_L_params.append(param)
                elif self.train_sp_L == False:
                    continue
                else:
                    raise ValueError(f"{self.train_sp_L} is not supported")
        

    def forward(self, x, t, aux):
        """
        Forward evaluation. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
            
            Parameters: 
                x : Tensor float[batch_size, max_nnomdes, in_dim] 
                    Input data
                t : Tensor float[batch_size, max_nnomdes, 1]
                    Temporal variable indicating the forecasting time horizon
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]  
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]  
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]  
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]  
                                     direted edge pairs; padding with 0  
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim] 
                                                 pinvdx on each directed edge 
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)] 

            
            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim] 
                       Input data

        """
        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        bases_c,  bases_s,  bases_0  = compute_Fourier_bases(nodes, self.modes * self.sp_L)
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)
        
        x_value = x[:,:,:self.ndims] 
        x = torch.cat((x,t),-1)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(self.softsign(compute_gradient(x, directed_edges, edge_gradient_weights)))
            x = x1 + x2 + x3
            #x = x1 + x2
            if self.act is not None and i != length - 1:
                x = self.act(x) 

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

       
        return x_value + t*x



# x_train, y_train, x_test, y_test are [n_data, n_x, n_channel] arrays
def TD_PCNO_train(x_train, t_train, aux_train, y_train, x_test, t_test, aux_test, y_test, config, model, save_model_name="./PCNO_model"):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    train_rel_l2_losses = []
    test_rel_l2_losses = []
    test_l2_losses = []
    normalization_x, normalization_y = config["train"]["normalization_x"], config["train"]["normalization_y"]
    normalization_dim_x, normalization_dim_y = config["train"]["normalization_dim_x"], config["train"]["normalization_dim_y"]
    non_normalized_dim_x, non_normalized_dim_y = config["train"]["non_normalized_dim_x"], config["train"]["non_normalized_dim_y"]
    
    ndims = model.ndims # n_train, size, n_channel
    print("In PCNO_train, ndims = ", ndims)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if normalization_x:
        x_normalizer = UnitGaussianNormalizer(x_train, non_normalized_dim = non_normalized_dim_x, normalization_dim=normalization_dim_x)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_normalizer.to(device)
        
    if normalization_y:
        y_normalizer = UnitGaussianNormalizer(y_train, non_normalized_dim = non_normalized_dim_y, normalization_dim=normalization_dim_y)
        y_train = y_normalizer.encode(y_train)
        y_test = y_normalizer.encode(y_test)
        y_normalizer.to(device)


    node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train = aux_train
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, t_train, y_train, node_mask_train, nodes_train, node_weights_train, directed_edges_train, edge_gradient_weights_train), 
                                               batch_size=config['train']['batch_size'], shuffle=True)
    
    node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test = aux_test
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, t_test, y_test, node_mask_test, nodes_test, node_weights_test, directed_edges_test, edge_gradient_weights_test), 
                                               batch_size=config['train']['batch_size'], shuffle=False)
    
    
    # Load from checkpoint
    # optimizer = Adam(model.parameters(), betas=(0.9, 0.999),
    #                  lr=config['train']['base_lr'], weight_decay=config['train']['weight_decay'])
    
    # if config['train']['scheduler'] == "MultiStepLR":
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=config['train']['milestones'],
    #                                                  gamma=config['train']['scheduler_gamma'])
    # elif config['train']['scheduler'] == "CosineAnnealingLR":
    #     T_max = (config['train']['epochs']//10)*(n_train//config['train']['batch_size'])
    #     eta_min  = 0.0
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min = eta_min)
    # elif config["train"]["scheduler"] == "OneCycleLR":
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer, max_lr=config['train']['base_lr'], 
    #         div_factor=2, final_div_factor=100,pct_start=0.2,
    #         steps_per_epoch=1, epochs=config['train']['epochs'])
    # else:
    #     print("Scheduler ", config['train']['scheduler'], " has not implemented.")
    optimizer = CombinedOptimizer(model.normal_params,model.sp_L_params,
        betas=(0.9, 0.999),
        lr=config["train"]["base_lr"],
        lr_ratio = config["train"]["lr_ratio"],
        weight_decay=config["train"]["weight_decay"],
        )
    
    scheduler = Combinedscheduler_OneCycleLR(
        optimizer, max_lr=config['train']['base_lr'], lr_ratio = config["train"]["lr_ratio"],
        div_factor=2, final_div_factor=100,pct_start=0.2,
        steps_per_epoch=1, epochs=config['train']['epochs'])
    
    model.train()
    myloss = LpLoss(d=1, p=2, size_average=False)

    epochs = config['train']['epochs']


    for ep in range(epochs):
        t1 = default_timer()
        train_rel_l2 = 0

        model.train()
        for x, t, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in train_loader:
            x, t, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device),t.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

            batch_size_ = x.shape[0]
            optimizer.zero_grad()
            out = model(x, t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)
            if normalization_y:
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
            out = out * node_mask  #mask the padded value with 0,(1 for node, 0 for padding)
            

            eps = torch.rand(1)# random time step
            eps = eps.to(device)
            temp_out = model(x, eps*t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1), Semigroup constraint intermediate variables
            if normalization_y:
                temp_out = y_normalizer.decode(temp_out)
            temp_out = temp_out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
            temp_x = torch.cat((temp_out, x[:,:,ndims:]),-1)
            if normalization_x:
                temp_x = x_normalizer.encode(temp_x)

            
            final_out = model(temp_x, (1-eps)*t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1),Semigroup constraint final variables
            if normalization_y:
                final_out = y_normalizer.decode(final_out)
            final_out = final_out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)


            loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)) + myloss(out.view(batch_size_,-1), final_out.view(batch_size_,-1))
            loss.backward()

            optimizer.step()
            train_rel_l2 += loss.item()

        test_l2 = 0
        test_rel_l2 = 0
        with torch.no_grad():
            for x, t, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights in test_loader:
                x, t, y, node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = x.to(device), t.to(device), y.to(device), node_mask.to(device), nodes.to(device), node_weights.to(device), directed_edges.to(device), edge_gradient_weights.to(device)

                batch_size_ = x.shape[0]
                out = model(x, t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1)

                if normalization_y:
                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(y)
                out=out*node_mask #mask the padded value with 0,(1 for node, 0 for padding)

                eps = torch.rand(1) #random time step
                eps = eps.to(device)
                temp_out = model(x, eps*t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1),Semigroup constraint intermediate variables
                if normalization_y:
                    temp_out = y_normalizer.decode(temp_out)
                temp_out = temp_out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)
                
                temp_x = torch.cat((temp_out ,x[:,:,ndims:]),-1)
                if normalization_x:
                    temp_x = x_normalizer.encode(temp_x)
                final_out = model(temp_x, (1-eps)*t, (node_mask, nodes, node_weights, directed_edges, edge_gradient_weights)) #.reshape(batch_size_,  -1),Semigroup constraint final variables
                if normalization_y:
                    final_out = y_normalizer.decode(final_out)
                final_out = final_out * node_mask #mask the padded value with 0,(1 for node, 0 for padding)

                test_rel_l2 += myloss(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()
                test_l2 += myloss.abs(out.view(batch_size_,-1), y.view(batch_size_,-1)).item()




        scheduler.step()

        train_rel_l2/= n_train
        test_l2 /= n_test
        test_rel_l2/= n_test
        
        train_rel_l2_losses.append(train_rel_l2)
        test_rel_l2_losses.append(test_rel_l2)
        test_l2_losses.append(test_l2)
    

        t2 = default_timer()
        print("Epoch : ", ep, " Time: ", round(t2-t1,3), " Rel. Train L2 Loss : ", train_rel_l2, " Rel. Test L2 Loss : ", test_rel_l2, " Test L2 Loss : ", test_l2,
              ' 1/sp_L: ',[round(float(x[0]), 3) for x in model.sp_L.cpu().tolist()],
                  flush=True)
        
        if (ep %100 == 99) or (ep == epochs -1):    
            torch.save(model.state_dict(), save_model_name + ".pth")
    
    
    return train_rel_l2_losses, test_rel_l2_losses, test_l2_losses
