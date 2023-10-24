### uncertainty-based weighting schemes:

def unc_loss_E( probs, loss_pix ):
    unc_measure = torch.sum(-probs * torch.log(probs+1e-10), dim=0)
    unc_measure = torch.div(unc_measure.clone(), torch.log(torch.tensor(probs.shape[0])))
    unc_metric = torch.exp(unc_measure)
    loss_uw = unc_metric * loss_pix
    return loss_uw
        
def unc_loss_M( probs, loss_pix ):
    largest = torch.topk(probs, 2, dim=0).values
    unc_measure = 1 - largest[0,:,:] + largest[1,:,:]
    unc_metric = torch.exp(unc_measure)
    loss_uw = unc_metric * loss_pix
    return loss_uw
    
def unc_loss_D( probs, loss_pix ):
    sorted, _ = torch.sort(probs, dim=0, descending=True)
    unc_measure = 1 - sorted[0,:,:] + sorted[-1,:,:]
    unc_metric = torch.exp(unc_measure)
    loss_uw = unc_metric * loss_pix
    return loss_uw

def unc_loss_Mbar( probs, loss_pix ):
    sorted, _ = torch.sort(probs, dim=0, descending=True)
    unc_measure = 1 - torch.sum(sorted[0]-sorted,0)/(sorted.shape[0]-1)
    unc_metric = torch.exp(unc_measure)
    loss_uw = unc_metric * loss_pix
    return loss_uw

def unc_loss_Z( pred, label, loss_pix ):
    loss_pix[0,torch.logical_and(torch.argmax(pred[0],0)!=label[0], torch.max(pred[0],0)[0]>0.75)] = 0
    return loss_pix


### perturbation of a reduced number of pixels:

def unc_attack_pix(cfg):

    cfg_patch_opt = cfg['adv_patch']['optimization']

    cfg["device"]["gpu"] = 0
    torch.cuda.set_device(cfg["device"]["gpu"])
    
    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    train_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["train_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"],
        std_version = cfg["data"]["std_version"], 
        bottom_crop = 0
    )

    num_train_samples = cfg_patch_opt['num_opt_samples']
    if num_train_samples is not None:
        opt_loader, _ = torch.utils.data.random_split(
            train_loader, 
            [num_train_samples, len(train_loader)-num_train_samples])
        
    else:
        opt_loader = train_loader
    print("num optimization images (from training set): " + str(len(opt_loader)))

    validation_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg["data"]["val_split"],
        version= cfg["data"]["version"],
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
        img_norm = cfg["data"]["img_norm"],
        bgr = cfg["data"]["bgr"], 
        std_version = cfg["data"]["std_version"],
        bottom_crop = 0
    )

    n_classes = train_loader.n_classes

    valloader = data.DataLoader(
        validation_loader, 
        batch_size=cfg_patch_opt["batch_size_val"], 
        num_workers=cfg["device"]["n_workers"],
        shuffle=False
    )

    test_patch.test_unc_attack(
                cfg = cfg,
                loader = valloader,
                n_classes = n_classes
    )


def test_unc_attack( cfg, 
                loader,
                n_classes):
    
    device = torch.device("cuda")
    torch.cuda.set_device(cfg["device"]["gpu"])

    # Setup Model and patch
    model_file_name = os.path.split(cfg["model"]["path"])[1]
    model_name = model_file_name[: model_file_name.find("_")]
    print(model_name)
    model_dict = {"arch": cfg["model"]["arch"]}
    model = get_model(model_dict, n_classes, version=cfg["data"]["dataset"])
    state = torch.load(cfg["model"]["path"], map_location = 'cpu')
    state = get_model_state(state, model_name)
    model.load_state_dict(state)    

    model.eval()
    model.to(device)

    save_path_probs_adv = '/home/user/probs/'
    if not os.path.exists(save_path_probs_adv):
        os.makedirs(save_path_probs_adv)
    
    alpha=1
    eps=16
    num_it = min(int(eps+4), int(1.25*eps))
    eps = eps / 255
    alpha = alpha / 255
    loss = nn.CrossEntropyLoss(ignore_index=250,reduction='none')
    H, W = 1024, 2048

    for i, (images, labels, filenames) in enumerate(loader):
        print(filenames)

        images = images.to(device)
        labels = labels.to(device)
        if isinstance(labels, list):
            labels, extrinsic, intrinsic = labels
            extrinsic, intrinsic = extrinsic.to(device), intrinsic.to(device)
        
        with torch.no_grad():
            pred = model(images)
            torch.manual_seed(0)
            unc = torch.randn(1024,2048)
            unc[unc<0] *=-1
            unc[labels[0]==250] = 0
            sorted, _ = torch.sort(torch.flatten(unc), descending=True)
            th_val = sorted[int(600*300)]

        adv_img = images
        adv_img.requires_grad = True

        tbar=trange(num_it)
        for i in tbar:

            model.zero_grad()
            pred = model(adv_img)

            J_cls = loss(pred, labels.detach()) 
            J_cls[0,torch.logical_and(torch.argmax(pred[0],0)!=labels[0], torch.max(pred[0],0)[0]>0.75)] = 0
            J_ss = torch.sum( J_cls[0] ) / (H * W)
            J_ss.backward()
            
            im_grad = adv_img.grad

            noise = (alpha * torch.sign(im_grad)).clamp(-eps,eps)

            noise[0,0,unc<=th_val] = 0
            noise[0,1,unc<=th_val] = 0
            noise[0,2,unc<=th_val] = 0

            adv_img = (adv_img + noise).clamp(images-eps,images+eps)
            adv_img = Variable(adv_img, requires_grad=True)

        with torch.no_grad():
            ex_adv_out = model(adv_img)

        ex_adv_out_j = F.interpolate(ex_adv_out[0].unsqueeze(0), size=(512,1024), mode='bilinear', align_corners=True).squeeze(0)

        print(ex_adv_out_j.size())

        ex_adv_out_j = torch.softmax(ex_adv_out_j,0).cpu().data.numpy()
        np.save(save_path_probs_adv + filenames[0] + '.npy', ex_adv_out_j.astype('float16'))