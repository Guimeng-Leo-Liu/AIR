import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import numpy as np
import pickle
import math
import clip
from PIL import Image
from sklearn.decomposition import PCA

from utils.text_templates import imagenet_templates, part_templates, imagenet_templates_small
from utils.label_templates import imagenet_labels
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

# from mapper import latent_mappers

def hook_fn(module, grad_input, grad_output):
    print("Grad input:", grad_input)
    print("Grad output:", grad_output)

def circular_generator(iterable):
    while True:
        for item in iterable:
            yield item

class TorchPCA(torch.nn.Module):
    def __init__(self, sklearn_pca):
        super(TorchPCA, self).__init__()
        # Extract the components and mean from sklearn's PCA
        # and convert them to tensors
        components = torch.tensor(sklearn_pca.components_)
        mean = torch.tensor(sklearn_pca.mean_)

        # Create buffers so PyTorch doesn't track gradients
        self.register_buffer('components', components)
        self.register_buffer('mean', mean)

    def forward(self, x):
        # Subtract the PCA mean and perform linear transformation
        return (x - self.mean) @ self.components.T

class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, args, device, patch_loss_type='mae', direction_loss_type='cosine', clip_model='ViT-B/32'):
        super(CLIPLoss, self).__init__()

        self.args = args
        if self.args.clip_models:
            clip_model = self.args.clip_models[0]
        self.device = device
        self.clip_model_name = clip_model.replace('/', '')
        model, clip_preprocess = clip.load(clip_model, device=self.device)
        self.model = model.float()

        # print('CLIP model: \n')
        # for name, module in self.model.named_children():
        #     print(name, module)

        # finetuned = './path/to/ViT-B16_temp0.1.pth'
        # finetuned = None
        # if finetuned:
        #     self.model.load_state_dict(torch.load(finetuned))
        #     self.clip_model_name = finetuned.split('/')[3][:-4]
        self.n_ctx = None
        self.n_dim = None

        self.norm_after_mean = True
        self.gt = args.gt

        self.onedir = args.onedir

        if args.oneshot:
            image_name = os.listdir(args.oneshot)[0]
            self.oneshot = os.path.join(args.oneshot, image_name)
        else:
            self.oneshot = None


        if args.tenshot:
            # self.tenshot = []
            # image_name = os.listdir(args.tenshot)
            # for n in image_name:
            #     self.tenshot.append(os.path.join(args.tenshot, n))

            self.tenshot = []
            image_name = os.listdir(args.tenshot)
            for n in image_name:
                self.tenshot.append(os.path.join(args.tenshot, n))
        else:
            self.tenshot = None

        if args.pca and self.norm_after_mean:
            with open('./visualized_figure_{}/pca.pkl'.format(self.clip_model_name), 'rb') as f:
                self.pca = TorchPCA(pickle.load(f)).to(self.device)
        else:
            self.pca = None

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor
        self.global_target         = None
        self.gt_direction          = None
        self.adapted_direction = None
        self.target_direction      = None
        self.first_target_direction= None
        self.patch_text_directions = None
        self.domain_distance       = None
        self.dist_p                = None
        self.target_prototype      = None
        self.adapted_prototype     = None
        self.dis_txt_features      = None


        self.mapper                   = None
        self.source_tokenized_prompts = None
        self.target_tokenized_prompts = None
        self.source_prefix            = None
        self.source_suffix            = None
        self.target_prefix            = None
        self.target_suffix            = None
        self.modality_gap             = None
        self.tenshot_img_features     = torch.tensor([]).to(device)


        self.patch_loss     = DirectionLoss(patch_loss_type)
        self.direction_loss = DirectionLoss(direction_loss_type)
        self.patch_direction_loss = torch.nn.CosineSimilarity(dim=2)
        self.loss_img = torch.nn.CrossEntropyLoss()
        self.loss_txt = torch.nn.CrossEntropyLoss()

        self.lambda_global    = args.lambda_global
        self.lambda_patch     = args.lambda_patch
        self.lambda_direction = args.lambda_direction
        self.lambda_manifold  = args.lambda_manifold
        self.lambda_texture   = args.lambda_texture
        self.lambda_domain    = args.lambda_domain
        self.lambda_const     = args.lambda_const
        self.lambda_KL        = args.lambda_KL
        self.lambda_proto     = args.lambda_proto
        self.lambda_dis       = args.lambda_dis

        self.src_text_features = None
        self.target_text_features = None
        self.angle_loss = torch.nn.L1Loss()

        self.model_cnn, preprocess_cnn = clip.load("RN50", device=self.device)
        self.preprocess_cnn = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                        preprocess_cnn.transforms[:2] +                                                 # to match CLIP input scale assumptions
                                        preprocess_cnn.transforms[4:])                                                  # + skip convert PIL to tensor

        self.model.requires_grad_(False)
        self.model_cnn.requires_grad_(False)

        self.texture_loss = torch.nn.MSELoss()

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def encode_images_with_cnn(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess_cnn(images).to(self.device)
        return self.model_cnn.encode_image(images)
    
    def distance_with_templates(self, img: torch.Tensor, class_str: str, templates=imagenet_templates) -> torch.Tensor:

        text_features  = self.get_text_features(class_str, templates)
        image_features = self.get_image_features(img)

        similarity = 100 * image_features @ text_features.T # (2, 512) @ (512, 79) -> (2, 79)
        # print(image_features.shape, text_features.shape, similarity.shape)
        return similarity.mean(axis=1) # (2,)
        # similarity = image_features @ text_features.T
        #
        # return 1. - similarity
    
    def get_text_features(self, class_str: str, templates=imagenet_templates, norm: bool = True) -> torch.Tensor:
        template_text = self.compose_text_with_templates(class_str, templates)

        tokens = clip.tokenize(template_text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)

        if self.pca:
            source_features = self.pca(source_features)
            target_features = self.pca(target_features)

        # to be edited
        t_direction = target_features - source_features
        t_direction /= t_direction.clone().norm(dim=-1, keepdim=True)
        text_direction = t_direction.mean(axis=0, keepdim=True)
        text_direction /= text_direction.clone().norm(dim=-1, keepdim=True)

        return text_direction

    #################################### compute prompt ####################################
    def get_pre_suf_fix(self, ):
        self.n_dim = self.model.ln_final.weight.shape[0]
        if self.args.ctx_init != "":
            ctx_init = self.args.ctx_init.replace("_", " ")
            self.n_ctx = len(ctx_init.split(" "))
            prompt_prefix = ctx_init
        else:
            prompt_prefix = " ".join(["X"] * self.n_ctx)
        source_prompts = [prompt_prefix + " " + self.args.source_class]
        target_prompts = [prompt_prefix + " " + self.args.target_class]
        print("source prompts", source_prompts)
        print("target prompts", target_prompts)
        self.source_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in source_prompts]).to(self.device)
        self.target_tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in target_prompts]).to(self.device)
        source_embedding = self.model.token_embedding(self.source_tokenized_prompts).type(self.model.dtype)
        target_embedding = self.model.token_embedding(self.target_tokenized_prompts).type(self.model.dtype)
        self.source_prefix = source_embedding[:, :1, :].detach()  # sos
        self.source_suffix = source_embedding[:, 1 + self.n_ctx:, :].detach()  # eos = [cls + xxxxxx]
        self.target_prefix = target_embedding[:, :1, :].detach()
        self.target_suffix = target_embedding[:, 1 + self.n_ctx:, :].detach()

    def text_encoder(self, source_prompts, source_tokenized_prompts):
        x = source_prompts + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(0, 2, 1, 3)  # NLD -> LND
        for j in range(len(x)):
            x[j] = self.model.transformer(x[j])
        x = x.permute(0, 2, 1, 3)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)
        text_features = x[:, torch.arange(x.shape[1]),
                        source_tokenized_prompts.argmax(dim=-1)] @ self.model.text_projection

        return text_features

    def compute_text_features(self, prompts, source_prefix, source_suffix, source_tokenized_prompts, clip_model, batch):
        source_ctx = prompts.unsqueeze(1)
        source_prefix = source_prefix.expand(batch, -1, -1, -1)
        source_suffix = source_suffix.expand(batch, -1, -1, -1)
        source_prompts = torch.cat(
            [
                source_prefix,  # (batch, n_cls, 1, dim)
                source_ctx,  # (batch, n_cls, n_ctx, dim)
                source_suffix,  # (batch, n_cls, *, dim)
            ],
            dim=2,
        )
        text_features = self.text_encoder(source_prompts, source_tokenized_prompts)
        return text_features

    def get_prompts(self, w):
        if self.source_prefix is None:
            self.get_pre_suf_fix()
        if self.mapper is None:
            checkpoint_path = f"/{self.args.source_class}2{self.args.target_class}/mapper.pt"
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.mapper = latent_mappers.SingleMapper(self.args, self.n_dim)
            self.mapper.load_state_dict(checkpoint["m"], strict=True)
        self.mapper.eval()
        with torch.no_grad():
            prompts = torch.reshape(self.mapper(w[0]), (self.args.batch, self.args.n_ctx, self.n_dim)).type(self.model.dtype)
            source_delta_features = self.compute_text_features(prompts, self.source_prefix, self.source_suffix, self.source_tokenized_prompts,
                                                         self.model, self.args.batch)
            target_delta_features = self.compute_text_features(prompts, self.target_prefix, self.target_suffix, self.target_tokenized_prompts,
                                                         self.model, self.args.batch)
        return source_delta_features, target_delta_features

    def get_IPL_text(self, w):

        source_delta_features, target_delta_features = self.get_prompts(w)
        prompt_prefix = self.args.ctx_init + " {}." # "a photo of a {}"

        # E_T(["a photo of a [cls]"])
        init_source_features = self.get_text_features(self.args.source_class, templates=[prompt_prefix], norm=True)
        init_target_features = self.get_text_features(self.args.target_class, templates=[prompt_prefix], norm=True)

        # E_T(["X X X X [cls]"])
        delta_source_features = source_delta_features + init_source_features
        delta_target_features = target_delta_features + init_target_features
        text_source_features = delta_source_features / delta_source_features.clone().norm(dim=-1, keepdim=True)
        text_target_features = delta_target_features / delta_target_features.clone().norm(dim=-1, keepdim=True)
        text_source_features = text_source_features.squeeze(1)
        text_target_features = text_target_features.squeeze(1)

        return text_source_features, text_target_features

    def promptCL_loss(self, w, target_img, source_img=None, margin=0.001):
        if self.modality_gap is None:
            with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, 'gffhq'), 'rb') as f:
                source_img_features = pickle.load(f)
                source_img_features = torch.Tensor(source_img_features[0]).mean(axis=0, keepdim=True).to(self.device)
                if self.norm_after_mean:
                    source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)
            source_txt_features = self.get_text_features(self.args.source_class, templates=[self.args.ctx_init], norm=True)
            self.modality_gap = self.direction_loss(source_img_features, source_txt_features)
            print(f'modality gap = {self.modality_gap}')

        img_target_features = self.get_image_features(target_img)
        text_source_features, text_target_features = self.get_IPL_text(w)

        # print(img_target_features.shape, text_target_features.shape)
        logits_per_image = self.model.logit_scale.exp() * img_target_features @ text_target_features.T
        logits_per_text = torch.transpose(logits_per_image, dim0=0, dim1=1)
        ground_truth = torch.arange(len(img_target_features), dtype=torch.long, device=self.device)

        CL_loss = (self.loss_img(logits_per_image, ground_truth) + self.loss_txt(logits_per_text, ground_truth)) / 2



        # if source_img is None: # phase 1
        #     pos_similarity = self.direction_loss(img_target_features, text_target_features)
        #     neg_similarity = self.direction_loss(img_target_features, text_source_features)
        #     # triplet_loss = torch.relu(pos_similarity - neg_similarity + margin).mean() * 10.
        #     # pos_similarity = self.model.logit_scale.exp() * img_target_features @ text_target_features.T
        #     # neg_similarity = self.model.logit_scale.exp() * img_target_features @ text_source_features.T
        #     print('\n', neg_similarity - pos_similarity)
        #     triplet_loss = torch.nn.SoftMarginLoss()(neg_similarity - pos_similarity, torch.ones(len(img_target_features), dtype=torch.long, device=self.device))
        #
        # else: #phase 2
        #     img_source_features = self.get_image_features(source_img)
        #     pos_similarity = self.direction_loss(text_target_features, img_target_features)
        #     neg_similarity = self.direction_loss(text_target_features, img_source_features)
        #     # triplet_loss = torch.relu(pos_similarity - neg_similarity + margin).mean() * 100.
        #     # pos_similarity = self.model.logit_scale.exp() * text_target_features @ img_target_features.T
        #     # neg_similarity = self.model.logit_scale.exp() * text_target_features @ img_source_features.T
        #     print(neg_similarity - pos_similarity)
        #     triplet_loss = torch.nn.SoftMarginLoss()(neg_similarity - pos_similarity, torch.ones(len(img_target_features), dtype=torch.long, device=self.device))

        target_modality_gap = self.direction_loss(text_target_features, img_target_features) #pos_similarity
        modality_loss = 100. * torch.mean((target_modality_gap-self.modality_gap)**2)
        # triplet_loss = target_modality_gap.mean()
        print(CL_loss.item(), modality_loss.item())
        total_loss = CL_loss + modality_loss
        return total_loss
    #################################### compute prompt ####################################

    # def compute_img2img_direction(self, source_images: torch.Tensor, target_images: list) -> torch.Tensor:
    #     with torch.no_grad():
    #         src_encoding = self.get_image_features(source_images)
    #         src_encoding = src_encoding.mean(dim=0, keepdim=True)
    #
    #         target_encodings = []
    #         for target_img in target_images:
    #             preprocessed = self.clip_preprocess(Image.open(target_img)).unsqueeze(0).to(self.device)
    #
    #             encoding = self.model.encode_image(preprocessed)
    #             encoding /= encoding.clone().norm(dim=-1, keepdim=True)
    #
    #             target_encodings.append(encoding)
    #
    #         target_encoding = torch.cat(target_encodings, axis=0)
    #         target_encoding = target_encoding.mean(dim=0, keepdim=True)
    #
    #         direction = target_encoding - src_encoding
    #         direction /= direction.clone().norm(dim=-1, keepdim=True)
    #
    #     return direction

    def compute_img2img_direction(self, source_images: torch.Tensor, target_images: str) -> torch.Tensor:
        with torch.no_grad():
            src_encoding = source_images

            preprocessed = self.clip_preprocess(Image.open(target_images)).unsqueeze(0).to(self.device)
            target_encoding = self.model.encode_image(preprocessed)
            target_encoding /= target_encoding.clone().norm(dim=-1, keepdim=True)

            direction = target_encoding - src_encoding
            direction /= direction.clone().norm(dim=-1, keepdim=True)

        return direction

    def set_text_features(self, source_class: str, target_class: str) -> None:
        source_features = self.get_text_features(source_class).mean(axis=0, keepdim=True)
        self.src_text_features = source_features / source_features.norm(dim=-1, keepdim=True)

        target_features = self.get_text_features(target_class).mean(axis=0, keepdim=True)
        self.target_text_features = target_features / target_features.norm(dim=-1, keepdim=True)

    def clip_angle_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.src_text_features is None:
            self.set_text_features(source_class, target_class)

        cos_text_angle = self.target_text_features @ self.src_text_features.T
        text_angle = torch.acos(cos_text_angle)

        src_img_features = self.get_image_features(src_img).unsqueeze(2)
        target_img_features = self.get_image_features(target_img).unsqueeze(1)

        cos_img_angle = torch.clamp(target_img_features @ src_img_features, min=-1.0, max=1.0)
        img_angle = torch.acos(cos_img_angle)

        text_angle = text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)
        cos_text_angle = cos_text_angle.unsqueeze(0).repeat(img_angle.size()[0], 1, 1)

        return self.angle_loss(cos_img_angle, cos_text_angle)

    def compose_text_with_templates(self, text: str, templates=imagenet_templates) -> list:
        return [template.format(text) for template in templates]

    def img_specific_mse(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)
        mse_loss = self.direction_loss(src_encoding, target_encoding)
        return mse_loss

    def direction_loss_to_deltaT(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:

        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)
        if self.pca:
            src_encoding = self.pca(src_encoding)
            target_encoding = self.pca(target_encoding)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            if self.pca:
                target_encoding = self.pca(target_encoding)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        directional_loss = self.direction_loss(edit_direction, self.first_target_direction).mean()
        # print(f"directional_loss: {directional_loss:.4f}")
        return directional_loss


    def get_gt_dir(self, target_class):
        if self.gt_direction is None:
            with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, 'gffhq'), 'rb') as f:
                source_img_features = pickle.load(f)
                source_img_features = torch.Tensor(source_img_features[0]).mean(axis=0, keepdim=True).to(self.device)
                # normalized mean vector
                if self.norm_after_mean:
                    source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)
                if self.pca:
                    source_img_features = self.pca(source_img_features)
            # with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, target_class), 'rb') as f:
            #     target_img_features = pickle.load(f)
            #     target_img_features = torch.Tensor(target_img_features[0]).mean(axis=0, keepdim=True).to(self.device)
            #     # normalized mean vector
            #     if self.norm_after_mean:
            #         target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
            #     if self.pca:
            #         target_img_features = self.pca(target_img_features)
            # target_direction = target_img_features - source_img_features
            # target_direction /= target_direction.clone().norm(dim=-1, keepdim=True)
            # self.gt_direction = target_direction
            oneshot = f'./oneshot_data/{target_class}'
            image_name = os.listdir(oneshot)[0]
            oneshot = os.path.join(oneshot, image_name)
            self.gt_direction = self.compute_img2img_direction(source_img_features, oneshot)

    def update_direction(self, frozen_embeddings, adapted_embeddings):
        print('============Updating direction============')
        directions = adapted_embeddings - frozen_embeddings
        directions /= directions.clone().norm(dim=-1, keepdim=True)

        reconstructed_direction = torch.tensor(directions).mean(0, keepdim=True).to(self.device)
        reconstructed_direction /= reconstructed_direction.clone().norm(dim=-1, keepdim=True)
        print(self.direction_loss(self.first_target_direction.detach().cpu(), torch.tensor(directions)).mean(0))
        # self.target_direction = reconstructed_direction

        target_direction = self.first_target_direction + self.args.adaptative_lambda * reconstructed_direction
        self.target_direction = target_direction / target_direction.clone().norm(dim=-1, keepdim=True)
        print(self.direction_loss(self.first_target_direction, self.target_direction))
        ##################################################################################################
        # directions = adapted_embeddings - frozen_embeddings
        # pca = PCA(n_components=512)
        # pca.fit(directions)
        #
        # reconstructed_direction = pca.inverse_transform(pca.transform(directions))
        # # print(type(reconstructed_direction))
        # reconstructed_direction = torch.tensor(reconstructed_direction).mean(0, keepdim=True).to(self.device)
        # reconstructed_direction /= reconstructed_direction.clone().norm(dim=-1, keepdim=True)
        # print(self.direction_loss(self.first_target_direction, reconstructed_direction))
        # # self.target_direction = reconstructed_direction
        #
        # target_direction = (1.0 - lambda_u) * self.first_target_direction + self.args.adaptative_lambda * reconstructed_direction
        # self.target_direction = target_direction / target_direction.clone().norm(dim=-1, keepdim=True)

        ##################################################################################################
        # directions = adapted_embeddings - frozen_embeddings
        # pca = PCA(n_components=512)
        # pca.fit(directions)
        #
        # target_direction = self.target_direction.detach().cpu()
        # reconstructed_direction = pca.inverse_transform(pca.transform(target_direction))
        # # print(type(reconstructed_direction))
        # reconstructed_direction = torch.tensor(reconstructed_direction).to(self.device)
        # reconstructed_direction /= reconstructed_direction.clone().norm(dim=-1, keepdim=True)
        # print(self.direction_loss(self.first_target_direction, reconstructed_direction))
        # # self.target_direction = reconstructed_direction
        #
        # target_direction = (1.0 - lambda_u) * self.first_target_direction + self.args.adaptative_lambda * reconstructed_direction
        # self.target_direction = target_direction / target_direction.clone().norm(dim=-1, keepdim=True)
        # print(self.direction_loss(self.first_target_direction, self.target_direction))





    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.target_direction is None:
            # # original NADA loss
            if not self.gt:
                if self.onedir:
                    print('============Using one pre-computed direction============')
                    with open('./prompt_features/{}2{}_mean.pkl'.format(self.args.source_class, self.args.target_class),
                              'rb') as f:
                        target_direction = pickle.load(f)
                        self.target_direction = target_direction.to(self.device)
                else:
                    print('============Using text direction============')
                    # print(f'{source_class=}')
                    # print(f'{target_class=}')
                    self.target_direction = self.compute_text_direction(source_class, target_class)

            else:
                if self.oneshot:
                    print('============Using one shot direction============')
                    with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, self.args.source_model_type), 'rb') as f:
                        source_img_features = pickle.load(f)
                        if isinstance(source_img_features, list):
                            source_img_features = source_img_features[0]
                        source_img_features = torch.Tensor(source_img_features).mean(axis=0,keepdim=True).to(self.device)
                        # normalized mean vector
                        if self.norm_after_mean:
                            source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)

                    self.target_direction = self.compute_img2img_direction(source_img_features, self.oneshot)

                elif self.tenshot:
                    with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, self.args.source_model_type), 'rb') as f:
                        source_img_features = pickle.load(f)
                        if isinstance(source_img_features, list):
                            source_img_features = source_img_features[0]
                        source_img_features = torch.Tensor(source_img_features).mean(axis=0,keepdim=True).to(self.device)
                        # normalized mean vector
                        if self.norm_after_mean:
                            source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)

                    if self.tenshot_img_features.numel() == 0:
                        with torch.no_grad():
                            for image_path in self.tenshot:
                                preprocessed = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                                target_img_features = self.model.encode_image(preprocessed)
                                target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
                                self.tenshot_img_features = torch.cat((self.tenshot_img_features, target_img_features))
                    target_img_features = self.tenshot_img_features.mean(0, keepdim=True).to(self.device)
                    if self.norm_after_mean:
                        target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
                    self.target_direction = target_img_features - source_img_features
                else:
                    print('============Using mean image direction============')
                    # gt image direction interpolation
                    with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, self.args.source_model_type), 'rb') as f:
                        source_img_features = pickle.load(f)
                        if isinstance(source_img_features, list):
                            source_img_features = source_img_features[0]
                        source_img_features = torch.Tensor(source_img_features).mean(axis=0,keepdim=True).to(self.device)
                        # normalized mean vector
                        if self.norm_after_mean:
                            source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)
                        if self.pca:
                            source_img_features = self.pca(source_img_features)
                    with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, f'r{target_class}'), 'rb') as f:
                        target_img_features1 = pickle.load(f)
                        if isinstance(target_img_features1, list):
                            target_img_features1 = target_img_features1[0]
                        target_img_features1 = torch.Tensor(target_img_features1).mean(axis=0, keepdim=True).to(self.device)
                        # normalized mean vector
                        if self.norm_after_mean:
                            target_img_features1 /= target_img_features1.clone().norm(dim=-1, keepdim=True)
                        if self.pca:
                            target_img_features1 = self.pca(target_img_features1)
                    with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, 'rcat'), 'rb') as f:
                        target_img_features2 = pickle.load(f)
                        if isinstance(target_img_features2, list):
                            target_img_features2 = target_img_features2[0]
                        target_img_features2 = torch.Tensor(target_img_features2).mean(axis=0, keepdim=True).to(self.device)
                        # normalized mean vector
                        if self.norm_after_mean:
                            target_img_features2 /= target_img_features2.clone().norm(dim=-1, keepdim=True)
                        if self.pca:
                            target_img_features2 = self.pca(target_img_features2)
                    #
                    target_direction1 = target_img_features1 - source_img_features
                    target_direction1 /= target_direction1.clone().norm(dim=-1, keepdim=True)
                    target_direction2 = target_img_features2 - source_img_features
                    target_direction2 /= target_direction2.clone().norm(dim=-1, keepdim=True)
                    #
                    weight = 0.0
                    self.target_direction = (1.0 - weight) * target_direction1 + weight * target_direction2
            self.first_target_direction = self.target_direction.clone()
            # print(self.target_direction)
        # print(self.direction_loss(self.first_target_direction, self.target_direction))
        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)
        if self.pca:
            src_encoding = self.pca(src_encoding)
            target_encoding = self.pca(target_encoding)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            if self.pca:
                target_encoding = self.pca(target_encoding)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
        directional_loss = self.direction_loss(edit_direction, self.target_direction).mean()
        # print(f"directional_loss: {directional_loss:.4f}")
        return directional_loss

    def mean_global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]

        D_cos = torch.stack([self.distance_with_templates(img, txt) for txt in text]) # (2, 2) [[i1t1, i2t1],[i1t2, i2t2]
        # print(D_cos.softmax(0).mean(1))

        # tokens = clip.tokenize(text).to(self.device)
        # image = self.preprocess(img)
        #
        # logits_per_image, _ = self.model(image, tokens)
        # if len(text) > 1:
        #     return logits_per_image.softmax(1)
        # # return cos distance

        return D_cos.softmax(0).mean(1)

    def eval_global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        tokens = clip.tokenize(["a "+txt for txt in text]).to(self.device)
        with torch.no_grad():
            image_features = self.encode_images(img)
            text_features = self.encode_text(tokens)

        text_features /= text_features.clone().norm(dim=-1, keepdim=True)
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        probs = self.model.logit_scale.exp() * image_features @ text_features.T # [batch, 512] @ [2, 512].T = [2,2]
        # print(probs.softmax(1).mean(axis=0))

        return probs.softmax(1).mean(axis=0)

    def discriminate_loss(self, img: torch.Tensor, src_cls, trg_cls, templates = imagenet_labels) -> torch.Tensor:
        if self.dis_txt_features is None:
            if src_cls not in templates:
                templates.append(src_cls)
            if trg_cls in templates:
                templates.remove(trg_cls)
            templates.append(trg_cls)
            # if not isinstance(text, list):
            #     text = [text]
            # [src_cls1, src_cls2, ...,  target_cls]
            tokens = clip.tokenize(["a "+txt for txt in templates]).to(self.device)
            with torch.no_grad():
                text_features = self.encode_text(tokens)
                text_features /= text_features.clone().norm(dim=-1, keepdim=True)
            self.dis_txt_features = text_features
            del templates, tokens

        image_features = self.encode_images(img)
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        probs = self.model.logit_scale.exp() * image_features @ self.dis_txt_features.T # [batch, 512] @ [2, 512].T = [batch,2]
        # print(probs.softmax(1).mean(axis=0))
        P_target = probs.softmax(1)[:, -1] # the probability belong to target cls [0, 1]
        P_target = torch.clamp(P_target, min=1e-7, max=1 - 1e-7) # [1e-7, 1 - 1e-7]
        Dis_loss = torch.nn.Softplus()(1.0-P_target).mean() - torch.log(torch.tensor(2.0))
        # Dis_loss = -torch.log(P_target).mean()

        return Dis_loss

    def adapted_distance_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        assert self.gt or self.oneshot
        if not self.domain_distance:
            print('============Domain distance============')
            with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, 'gffhq'), 'rb') as f:
                source_img_features = pickle.load(f)
                source_img_features = torch.Tensor(source_img_features[0]).mean(axis=0, keepdim=True).to(self.device)
                # normalized mean vector
                if self.norm_after_mean:
                    source_img_features /= source_img_features.clone().norm(dim=-1, keepdim=True)

            with torch.no_grad():
                preprocessed = self.clip_preprocess(Image.open(self.oneshot)).unsqueeze(0).to(self.device)
                target_img_features = self.model.encode_image(preprocessed)
                target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
            self.domain_distance = 1.0 - torch.nn.CosineSimilarity()(target_img_features, source_img_features)
            print(f'Domain distance = {self.domain_distance.item():3f}')
        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        adapted_distance = 1.0 - torch.nn.CosineSimilarity()(target_encoding, src_encoding)

        if adapted_distance == 0:
            target_encoding = self.get_image_features(target_img + 1e-6)
            adapted_distance = 1.0 - torch.nn.CosineSimilarity()(target_encoding, src_encoding)

        domain_loss = torch.abs((self.domain_distance - adapted_distance).mean(axis=0))
        # print(f"dd_loss: {domain_loss:.4f}")
        return domain_loss

    def const_loss(self, src_img: torch.Tensor, target_img: torch.Tensor):
        src_encoding = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)

        target_sim = []
        src_sim = []
        for i in range(target_encoding.size(0)):
            target_temp = 1.0 - torch.nn.CosineSimilarity()(target_encoding[i].unsqueeze(0), target_encoding)
            src_temp = 1.0 - torch.nn.CosineSimilarity()(src_encoding[i].unsqueeze(0), src_encoding)
            for j in range(target_encoding.size(0)):
                if i != j:
                    target_sim.append(target_temp[j].unsqueeze(0))
                    src_sim.append(src_temp[j].unsqueeze(0))

        target_sim = torch.cat(target_sim, dim=0)
        src_sim = torch.cat(src_sim, dim=0)

        loss_co2 = torch.mean((target_sim - src_sim) ** 2)
        # return torch.max(loss_co2-0.15, torch.tensor(0))
        return loss_co2

    def KL_loss(self, img):
        if not self.dist_p:
            print('============KL Divergence============')
            with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, self.args.source_model_type), 'rb') as f:
                target_img_features = pickle.load(f)
                target_img_features = torch.Tensor(target_img_features[0])
                mu_p = target_img_features.mean(0, keepdim=True)
                # if self.norm_after_mean:
                #     mu_p /= mu_p.clone().norm(dim=-1, keepdim=True)
                centered_feature = target_img_features - mu_p
                cov_p = centered_feature.T @ centered_feature / centered_feature.size(0)
                self.dist_p = MultivariateNormal(mu_p.float().to(self.device), cov_p.float().to(self.device))

        image_features = self.encode_images(img)
        image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        mu_q = image_features.mean(0, keepdim=True)
        # if self.norm_after_mean:
        #     mu_q /= mu_q.clone().norm(dim=-1, keepdim=True)
        centered_img_feature = image_features - mu_q
        cov_q = centered_img_feature.T @ centered_img_feature / centered_img_feature.size(0)
        cov_q += torch.eye(cov_q.size(0)).to(self.device) * 1e-6
        cov_q = (cov_q + cov_q.t()) / 2
        print(torch.isnan(cov_q).any())
        print(torch.isinf(cov_q).any())
        dist_q = MultivariateNormal(mu_q.float(), cov_q.float())

        kl_result = kl_divergence(self.dist_p, dist_q)
        return kl_result

    def get_global_target(self, text):
        oneshot = f'./oneshot_data/{text}'
        image_name = os.listdir(oneshot)[0]
        oneshot = os.path.join(oneshot, image_name)
        with torch.no_grad():
            preprocessed = self.clip_preprocess(Image.open(oneshot)).unsqueeze(0).to(self.device)
            target_img_features = self.model.encode_image(preprocessed)
        target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
        self.global_target = target_img_features

    def global_clip_loss(self, img: torch.Tensor, text) -> torch.Tensor: ## 1-cos(E_I^train, E_I^target)
        if self.gt:
            if self.oneshot:
                with torch.no_grad():
                    preprocessed = self.clip_preprocess(Image.open(self.oneshot)).unsqueeze(0).to(self.device)
                    target_img_features = self.model.encode_image(preprocessed)
                    target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
            elif self.tenshot:
                if self.tenshot_img_features.numel() == 0:
                    with torch.no_grad():
                        for image_path in self.tenshot:
                            preprocessed = self.clip_preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
                            target_img_features = self.model.encode_image(preprocessed)
                            target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
                            self.tenshot_img_features = torch.cat((self.tenshot_img_features, target_img_features))

                target_img_features = self.tenshot_img_features.mean(0, keepdim=True)
                # normalized mean vector
                if self.norm_after_mean:
                    target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)

            else:
                with open('./visualized_figure_{}/{}_features.pkl'.format(self.clip_model_name, f'r{text}'), 'rb') as f:
                    target_img_features = pickle.load(f)
                    if isinstance(target_img_features, list):
                        target_img_features = target_img_features[0]
                    target_img_features = torch.Tensor(target_img_features).mean(0, keepdim=True).to(self.device)
                    # normalized mean vector
                    if self.norm_after_mean:
                        target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)

            image_features = self.encode_images(img)
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
            # target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)

            # if self.tenshot:
            #     target_img_features = torch.tensor([]).to(self.device)
            #     for feature in image_features:
            #         index = torch.argmin(self.direction_loss(feature, self.tenshot_img_features))
            #         target_img_features = torch.cat((target_img_features, self.tenshot_img_features[index].unsqueeze(0)))
            #         print(index.item())

            global_loss = self.direction_loss(image_features, target_img_features).mean()
            # logits_per_image = image_features.float() @ target_img_features.float().T
            # global_loss = (1. - logits_per_image).mean()
            # print(f"global_loss: {global_loss}")
            return global_loss
        else:
            if not isinstance(text, list):
                text = [text]

            tokens = clip.tokenize(text).to(self.device)
            image  = self.preprocess(img)

            logits_per_image, _ = self.model(image, tokens)
            # return cos distance
            global_loss = (1. - logits_per_image / 100).mean()
            # print(f"global_loss: {global_loss:.4f}")
            return global_loss

    def proto_loss(self, img: torch.Tensor) -> torch.Tensor: ## 1-cos(E_I^train, E_I^target)
        if self.target_prototype is None:
            assert self.gt
            if self.oneshot:
                with torch.no_grad():
                    preprocessed = self.clip_preprocess(Image.open(self.oneshot)).unsqueeze(0).to(self.device)
                    target_img_features = self.model.encode_image(preprocessed)
                    target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
                    self.target_prototype = target_img_features
            else:
                with open('./visualized_figure_{}/r{}_features.pkl'.format(self.clip_model_name, self.args.source_model_type), 'rb') as f:
                    target_img_features = pickle.load(f)
                    target_img_features = torch.Tensor(target_img_features[0]).mean(0, keepdim=True).to(self.device)
                    # normalized mean vector
                    if self.norm_after_mean:
                        target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)
                    self.target_prototype = target_img_features

        decay = 0.95
        image_features = self.get_image_features(img)
        mean_features = image_features.mean(0, keepdim=True)
        if self.norm_after_mean:
            mean_features /= mean_features.clone().norm(dim=-1, keepdim=True)

        if self.adapted_prototype is None:
            self.adapted_prototype = mean_features
        else:
            self.adapted_prototype = self.adapted_prototype.detach() * decay + mean_features * (1.0-decay)
        # print(self.adapted_prototype)
        # print(self.target_prototype)
        protoloss = 1.0 - torch.nn.CosineSimilarity()(self.adapted_prototype, self.target_prototype)
        return protoloss.squeeze(0)

    def random_patch_centers(self, img_shape, num_patches, size):
        batch_size, channels, height, width = img_shape

        half_size = size // 2
        patch_centers = np.concatenate([np.random.randint(half_size, width - half_size,  size=(batch_size * num_patches, 1)),
                                        np.random.randint(half_size, height - half_size, size=(batch_size * num_patches, 1))], axis=1)

        return patch_centers

    def generate_patches(self, img: torch.Tensor, patch_centers, size):
        batch_size  = img.shape[0]
        num_patches = len(patch_centers) // batch_size
        half_size   = size // 2

        patches = []

        for batch_idx in range(batch_size):
            for patch_idx in range(num_patches):

                center_x = patch_centers[batch_idx * num_patches + patch_idx][0]
                center_y = patch_centers[batch_idx * num_patches + patch_idx][1]

                patch = img[batch_idx:batch_idx+1, :, center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size]

                patches.append(patch)

        patches = torch.cat(patches, axis=0)

        return patches

    def patch_scores(self, img: torch.Tensor, class_str: str, patch_centers, patch_size: int) -> torch.Tensor:

        parts = self.compose_text_with_templates(class_str, part_templates)    
        tokens = clip.tokenize(parts).to(self.device)
        text_features = self.encode_text(tokens).detach()

        patches        = self.generate_patches(img, patch_centers, patch_size)
        image_features = self.get_image_features(patches)


        similarity = image_features @ text_features.T

        return similarity

    def clip_patch_similarity(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        patch_size = 196 #TODO remove magic number

        patch_centers = self.random_patch_centers(src_img.shape, 4, patch_size) #TODO remove magic number
   
        src_scores    = self.patch_scores(src_img, source_class, patch_centers, patch_size)
        target_scores = self.patch_scores(target_img, target_class, patch_centers, patch_size)

        return self.patch_loss(src_scores, target_scores)

    def patch_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:

        if self.patch_text_directions is None:
            src_part_classes = self.compose_text_with_templates(source_class, part_templates)
            target_part_classes = self.compose_text_with_templates(target_class, part_templates)

            parts_classes = list(zip(src_part_classes, target_part_classes))

            self.patch_text_directions = torch.cat([self.compute_text_direction(pair[0], pair[1]) for pair in parts_classes], dim=0)

        patch_size = 510 # TODO remove magic numbers

        patch_centers = self.random_patch_centers(src_img.shape, 1, patch_size)

        patches = self.generate_patches(src_img, patch_centers, patch_size)
        src_features = self.get_image_features(patches)

        patches = self.generate_patches(target_img, patch_centers, patch_size)
        target_features = self.get_image_features(patches)

        edit_direction = (target_features - src_features)
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)

        cosine_dists = 1. - self.patch_direction_loss(edit_direction.unsqueeze(1), self.patch_text_directions.unsqueeze(0))

        patch_class_scores = cosine_dists * (edit_direction @ self.patch_text_directions.T).softmax(dim=-1)

        return patch_class_scores.mean()

    def cnn_feature_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        src_features = self.encode_images_with_cnn(src_img)
        target_features = self.encode_images_with_cnn(target_img)

        return self.texture_loss(src_features, target_features)

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, texture_image: torch.Tensor = None, write=False):
        clip_loss = {}

        if self.lambda_global:
            clip_loss['global'] = self.lambda_global * self.global_clip_loss(target_img, [f"a {target_class}"])

        if self.lambda_patch:
            clip_loss['patch'] = self.lambda_patch * self.patch_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_direction:
            clip_loss['direction'] = self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)

        if self.lambda_manifold:
            clip_loss['manifold'] = self.lambda_manifold * self.clip_angle_loss(src_img, source_class, target_img, target_class)

        if self.lambda_texture and (texture_image is not None):
            clip_loss['texture'] = self.lambda_texture * self.cnn_feature_loss(texture_image, target_img)

        if self.lambda_domain:
            clip_loss['domain'] = self.lambda_domain * self.adapted_distance_loss(src_img, target_img)

        if self.lambda_const:
            clip_loss['const'] = self.lambda_const * self.const_loss(src_img, target_img)

        if self.lambda_KL:
            clip_loss['KL'] = self.lambda_KL * self.KL_loss(target_img)

        if self.lambda_proto:
            clip_loss['proto'] = self.lambda_proto * self.proto_loss(target_img)

        if self.lambda_dis:
            clip_loss['dis'] = self.lambda_dis * self.discriminate_loss(target_img, source_class, target_class, templates=imagenet_labels)

        if write:
            D_cos = self.eval_global_clip_loss(target_img, [source_class, target_class])
            return clip_loss, D_cos

        return clip_loss
