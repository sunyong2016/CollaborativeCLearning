import copy

import pytorch_lightning as pl
import torch
import torchvision
from lightly import loss, models
from lightly.models import utils
from lightly.models.modules import heads
from lightly.transforms.smog_transform import SMoGTransform
from sklearn.cluster import KMeans
from torch import nn

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices_num = 1

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')


class SMoGModel(nn.Module):

    def __init__(self,
                 backbone,
                 n_groups=300,
                 in_dim=512,
                 a_hid_dim=128,
                 b_hid_dim=2048,
                 beta=0.99):
        super().__init__()

        self.backbone = backbone
        self.projection_head = SMoGProjectionHead(in_dim, b_hid_dim, a_hid_dim)
        self.prediction_head = SMoGPredictionHead(a_hid_dim, b_hid_dim,
                                                  a_hid_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.n_groups = n_groups
        self.smog = SMoGPrototypes(group_features=torch.rand(
            self.n_groups, a_hid_dim),
                                   beta=beta)

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        # clusters the features using sklearn
        # (note: faiss is probably more efficient)
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        # see https://arxiv.org/pdf/2207.06167.pdf Table 7b)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

    def forward_momentum(self, x):
        features = self.backbone_momentum(x).flatten(start_dim=1)
        encoded = self.projection_head_momentum(features)
        return encoded


def create_data_loader(path_to_data, args, test_percent=0.0):
    image_mean, image_std = get_image_mean_std()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.input_size),
        torchvision.transforms.CenterCrop(args.input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(image_mean, image_std)
    ])

    rs_data = torchvision.datasets.ImageFolder(path_to_data, transform)
    num_classes = len(rs_data.classes)

    len_rs_data = len(rs_data)  # total number of examples
    num_test = int(test_percent * len_rs_data)  # take  10% for test
    num_train = len_rs_data - num_test  # take  80% for train

    train_rs_data, test_rs_data = torch.utils.data.random_split(
        rs_data, lengths=[num_train, num_test])
    print(len_rs_data, num_train, len(train_rs_data))

    train_data_loader = torch.utils.data.DataLoader(train_rs_data,
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 drop_last=True,
                                                 num_workers=args.nThreads)

    test_data_loader = torch.utils.data.DataLoader(test_rs_data,
                                                 batch_size=args.batch_size)
    return train_rs_data, test_rs_data, train_data_loader, test_data_loader


def get_image_mean_std():
    image_mean, image_std = [0.485, 0.456, 0.406], [0.229, 0.224,
                                                    0.225]  # 归一化0-1
    return image_mean, image_std


def get_backbone_from_torchvision(
        backbone_name="resnet50"):  # 可以写程序测试不同resnet得到结果
    if backbone_name == "resnet18":
        source_model = torchvision.models.resnet18()

    elif backbone_name == "resnet34":
        source_model = torchvision.models.resnet34()

    elif backbone_name == "resnet50":
        source_model = torchvision.models.resnet50()

    elif backbone_name == "resnet101":
        source_model = torchvision.models.resnet101()

    elif backbone_name == "resnet152":
        source_model = torchvision.models.resnet152()

    elif backbone_name == "vit_b_16":
        source_model = torchvision.models.vit_b_16()

    elif backbone_name == "vit_b_32":
        source_model = torchvision.models.vit_b_32()

    elif backbone_name == "vit_b_32":
        source_model = torchvision.models.vit_b_32()

    elif backbone_name == "vit_l_16":
        source_model = torchvision.models.vit_l_16()

    elif backbone_name == "vit_l_32":
        source_model = torchvision.models.vit_l_32()

    else:
        source_model = torchvision.models.vit_h_14()

    backbone = torch.nn.Sequential(*list(source_model.children())[:-1])

    return backbone


def get_backbone_from_CLIP(backbone_name="ViT-B/16"):  # model_names = ['RN50', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
    # clip.available_models()
    print('load pre-trained model')
    clipmodel, preprocess = clip.load(
        backbone_name, device=device,
        jit=False)  # loading the CLIP model based on ViT
    # clipmodel.to(device).eval()
    input_resolution = clipmodel.visual.input_resolution
    context_length = clipmodel.context_length
    vocab_size = clipmodel.vocab_size
    return clipmodel, preprocess


def smog_training(backbone, tr_data_loader, 
                  global_step=0,
                  n_epochs=10,
                  noise_factor=0.5, lr=0.01, each_iterations=300, gamma=0.5):  # noise_factor for adding noise to images
    model = SMoGModel(backbone)

    # memory bank because we reset the group features every 300 iterations
    memory_bank_size = each_iterations * args.batch_size
    memory_bank = MemoryBankModule(size=memory_bank_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    global_criterion = nn.CrossEntropyLoss()  # global loss criterion
    local_criterion = NTXentLoss()  # local loss criterion

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=1e-6)

    print("Starting Training")
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tr_data_loader):
            x0 = batch[0]
            ## add random noise to the input images
            noisy_imgs = x0 + noise_factor * torch.randn(*x0.shape)  #denoising
            # Clip the images to be between 0 and 1
            x1 = np.clip(noisy_imgs, 0., 1.)

            if batch_idx % 2:
                # swap batches every second iteration
                x1, x0 = x0, x1

            x0 = x0.to(device)
            x1 = x1.to(device)

            if global_step > 0 and global_step % each_iterations == 0:
                # reset group features and weights every 300 iterations
                model.reset_group_features(memory_bank=memory_bank)
                model.reset_momentum_weights()
            else:
                # update momentum
                utils.update_momentum(model.backbone, model.backbone_momentum,
                                      0.99)
                utils.update_momentum(model.projection_head,
                                      model.projection_head_momentum, 0.99)

            x0_encoded, x0_predicted = model(x0)
            x1_encoded = model.forward_momentum(x1)

            # update group features and get group assignments
            assignments = model.smog.assign_groups(x1_encoded)
            group_features = model.smog.get_updated_group_features(x0_encoded)
            logits = model.smog(x0_predicted, group_features, temperature=0.1)
            model.smog.set_group_features(group_features)

            loss = gamma * global_criterion(logits,
                                          assignments) + (1-gamma) * local_criterion(
                                              x0_encoded, x1_encoded)

            # use memory bank to periodically reset the group features with k-means
            memory_bank(x0_encoded, update=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            total_loss += loss.detach()

        avg_loss = total_loss / len(tr_data_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    return model
