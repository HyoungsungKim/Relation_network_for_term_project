from random import shuffle
import torch
from torch.cuda import is_available
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

import numpy as np
import task_generator_no_args as tg
import os
import math
import random

writer = SummaryWriter(logdir='scalar')
# * Hyper Parameters
# * Hyper Parameters for 5 way 5 shots train
FEATURE_DIM = 64  # args.feature_dim
RELATION_DIM = 8  # args.relation_dim
CLASS_NUM = 5  # args.class_num
SAMPLE_NUM_PER_CLASS = 5  # args.sample_num_per_class
BATCH_NUM_PER_CLASS = 15  # args.batch_num_per_class
EPISODE = 1000000  # args.episode
TEST_EPISODE = 1000  # args.test_episode
LEARNING_RATE = 0.001  # args.learning_rate
# GPU = # args.gpu
HIDDEN_UNIT = 10  # args.hidden_unit
ENTROPY_WEIGHT = 1e-2


class CNNEncoder(nn.Module):
    """
    Docstring for ClassName
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)           
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, shape):
        return shape.view(shape.size(0), -1)


class FeaturesWeightNetwork(nn.Module):
    def __init__(self):
        super(FeaturesWeightNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.layer(x)


class RelationNetwork(nn.Module):
    """
    docstring for RelationNetwork
    """
    
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer3 = nn.Sequential(
            Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )        
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        return out
    
    
def weights_init(m):    
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
       
        
def feature_weights_init(m):    
    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())
            
    
def main():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.omniglot_character_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = CNNEncoder()
    feature_weight = FeaturesWeightNetwork()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
    
    feature_encoder.apply(weights_init)
    feature_weight.apply(weights_init)
    relation_network.apply(weights_init)
    
    feature_encoder.to(device)
    feature_weight.to(device)
    relation_network.to(device)
    
    feature_encoder.train()
    feature_weight.train()
    relation_network.train()
        
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    
    feature_weight_optim = torch.optim.Adam(feature_weight.parameters(), lr=LEARNING_RATE)
    feature_weight_scheduler = StepLR(feature_weight_optim, step_size=100000, gamma=0.5)
    
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")
        
    # * Step 3: build graph
    print("Training...")
    
    last_accuracy = 0.0
    
    loss_array = []
    for episode in range(EPISODE):
        
        # * init dataset
        # * sample_dataloader is to obtain previous samples for compare
        # * batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])
        task = tg.OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
        batch_dataloader = tg.get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True, rotation=degrees)
        
        # * sample datas
        # samples, sample_labels = sample_dataloader.__iter__().next()
        # batches, batch_labels = batch_dataloader.__iter__().next()
        
        samples, sample_labels = next(iter(sample_dataloader))
        batches, batch_labels = next(iter(batch_dataloader))
        
        samples, sample_labels = samples.to(device), sample_labels.to(device)
        batches, batch_labels = batches.to(device), batch_labels.to(device)
                        
        # * calculates features
        sample_features = feature_encoder(samples)
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
        
        # * torch.mean works better
        sample_features = torch.mean(sample_features, 1).squeeze(1)
        batch_features = feature_encoder(batches)
        
        # * calculate relations
        # * each batch sample link to every samples to calculate relations
        # * to form a 100 * 128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
        
        weighted_relation_pairs = feature_weight(relation_pairs)
        relations = relation_network(weighted_relation_pairs).view(-1, CLASS_NUM)
        
        # dist = Categorical(relations)
        # selections = dist.sample()       
        # entropy = dist.entropy()
        
        # labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM, requires_grad=True).to(device).scatter_(1, selections.view(-1, 1), 1)
        one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM, requires_grad=True).to(device).scatter_(1, batch_labels.view(-1, 1), 1)
        
        mse = nn.MSELoss()
        # assert selections.shape == batch_labels.shape
        # batch_label_tensor = torch.tensor(batch_labels, dtype=torch.float, requires_grad=True).to(device)       
        
        # selections = selections.type(torch.float)
        loss = mse(relations, one_hot_labels) 
        
        feature_encoder.zero_grad()
        feature_weight.zero_grad()        
        relation_network.zero_grad()
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)  
        # torch.nn.utils.clip_grad_norm_(feature_weight.parameters(), 0.5)      
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        
        feature_encoder_optim.step()
        feature_weight_optim.step()
        relation_network_optim.step()
                
        feature_encoder_scheduler.step(episode)
        feature_weight_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        
        if (episode + 1) % 100 == 0:
           # print(relations[:5])
           # print(one_hot_labels[:5])
            print(f"episode : {episode+1}, loss : {loss.cpu().detach().numpy()}")
            loss_array.append(loss.cpu().detach().numpy())
            
        if (episode + 1) % 500 == 0:
            print("Testing...")
            total_reward = 0
            
            feature_encoder.eval()
            feature_weight.eval()
            relation_network.eval()
    
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
                test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test", shuffle=True, rotation=degrees)
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                    
                # * calculate features
                sample_features = feature_encoder(sample_images)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                test_features = feature_encoder(test_images)
                
                # * calculate relations
                # * each batch sample link to every samples to calculate relations
                # * to form a 100x128 matrix for relation network
                
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
                
                weighted_relation_pairs = feature_weight(relation_pairs)
                relations = relation_network(weighted_relation_pairs).view(-1, CLASS_NUM)
                
                _, predict_labels = torch.max(relations.data, 1)
                
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
                total_reward += np.sum(rewards)
                        
            feature_encoder.train()
            feature_weight.train()
            relation_network.train()   
            test_accuracy = total_reward / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS * TEST_EPISODE)
            
            mean_loss = np.mean(loss_array)
            print("test accuracy : ", test_accuracy)
            print(f'mean loss : {mean_loss}')
            writer.add_scalar('loss', mean_loss, episode + 1)
            writer.add_scalar('test accuracy', test_accuracy, episode + 1)
            loss_array = []
            
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(
                    feature_encoder.state_dict(),
                    str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                torch.save(
                    relation_network.state_dict(),
                    str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy    
    
            
if __name__ == "__main__":
    main()
