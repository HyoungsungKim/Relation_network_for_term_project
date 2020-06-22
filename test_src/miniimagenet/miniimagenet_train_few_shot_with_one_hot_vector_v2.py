import torch
from torch import embedding
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import numpy as np
import task_generator as tg
import os
import argparse
import random
writer = SummaryWriter(logdir='scalar')

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


class CNNEncoder(nn.Module):
    """
    Docstring for ClassName
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.Dropout2d()
        )
        
        self.layer4 = nn.Sequential(            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU()
        )
        
        conv_out_size = self._get_conv_out()        
        
        self.linear = nn.Sequential(            
            Flatten(),
            nn.Linear(conv_out_size, 8),            
            nn.LeakyReLU(),
            nn.Linear(8, SAMPLE_NUM_PER_CLASS),
           # nn.LeakyReLU()
           # nn.Softmax(dim=-1)
        )
        
    def _get_conv_out(self):       
        temp = torch.zeros([CLASS_NUM * SAMPLE_NUM_PER_CLASS, 3, 84, 84]) 
        o1 = self.layer1(temp)
        o2 = self.layer2(o1)
        o3 = self.layer3(o2)
        o4 = self.layer4(o3)
        
        conv_out = int(np.prod(o4.size()[1:]))
        # * e.g. conv_out : [25, 40000]
        return conv_out
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        cnn_result = out
        linear = self.linear(out)
        
        return cnn_result, linear


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, shape):
        return shape.view(shape.size(0), -1)


class RelationNetwork(nn.Module):
    """
    docstring for RelationNetwork
    """
    
    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )
        
        self.layer3 = nn.Sequential(
            Flatten(),
            nn.Linear(input_size * 3 * 3, hidden_size),
            nn.LeakyReLU(),
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
        
        
def main():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)
    
    feature_encoder.train()
    relation_network.train()
    
    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)
    
    feature_encoder.to(device)
    relation_network.to(device)
    
    mse = nn.MSELoss()
    cross_entropy = nn.CrossEntropyLoss()
    
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    # feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=100000, gamma=0.5)
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")
        
    # * Step 3: build graph
    print("Training...")
    
    last_accuracy = 0.0
    embedding_loss_list = []
    relation_loss_list = []
    loss_list = []
    for episode in range(EPISODE):
        
        # * init dataset
        # * sample_dataloader is to obtain previous samples for compare
        # * batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])
        task = tg.MiniImagenetTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
       # sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", rotation=degrees)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=True, rotation=degrees)
        
        # * sample datas
        # samples, sample_labels = sample_dataloader.__iter__().next()
        # batches, batch_labels = batch_dataloader.__iter__().next()
        
        samples, sample_labels = next(iter(sample_dataloader))
        batches, batch_labels = next(iter(batch_dataloader))
        
        samples, sample_labels = samples.to(device), sample_labels.to(device)
        batches, batch_labels = batches.to(device), batch_labels.to(device)
                        
        # one_hot_sample_labels = torch.zeros(SAMPLE_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).to(device, dtype=torch.long ).scatter_(1, sample_labels.view(-1, 1), 1)
        # one_hot_batch_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).to(device).scatter_(1, batch_labels.view(-1, 1), 1)
        
        # * calculates features
        sample_features, linear = feature_encoder(samples)
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        batch_features, batch_linear = feature_encoder(batches)
        
        # embedding_sample_loss = cross_entropy(linear, sample_labels)
        embedding_loss = cross_entropy(linear, sample_labels)
        # embedding_loss = cross_entropy(batch_linear, batch_labels)
        # embedding_loss = embedding_sample_loss + embedding_batch_loss        
        
        # * calculate relations
        # * each batch sample link to every samples to calculate relations
        # * to form a 100 * 128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                
        # one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).to(device).scatter_(1, batch_labels.view(-1, 1), 1)
        
        # relation_loss = mse(relations, one_hot_labels)
        relation_loss = cross_entropy(relations, batch_labels)
        loss = embedding_loss + relation_loss
        # loss = relation_loss
        
        # feature_encoder.zero_grad()
        feature_encoder_optim.zero_grad()        
        # relation_network.zero_grad()
        relation_network_optim.zero_grad()
        
        # embedding_loss.requires_grad = False
        # embedding_loss.backward(retain_graph=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        
        feature_encoder_optim.step()
        relation_network_optim.step()
                
        #feature_encoder_scheduler.step(episode)
        #relation_network_scheduler.step(episode)
        
        if (episode + 1) % 200 == 0:
            print(f"episode : {episode+1}, loss : {loss.cpu().detach().numpy()}")
            loss_list.append(loss.cpu().detach().numpy())
            embedding_loss_list.append(embedding_loss.cpu().detach().numpy())
            relation_loss_list.append(relation_loss.cpu().detach().numpy())
        if (episode + 1) % 1000 == 0:
            print("Testing...")
            total_reward = 0
            # print(feature_encoder.layer1.state_dict()['weight'])
            
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train")
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test")
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                    
                # * calculate features
                sample_features, _ = feature_encoder(sample_images)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                test_features, _ = feature_encoder(test_images)
                
                # * calculate relations
                # * each batch sample link to every samples to calculate relations
                # * to form a 100x128 matrix for relation network
                
                sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                
                _, predict_labels = torch.max(relations.data, 1)
                
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
                total_reward += np.sum(rewards)
                
            test_accuracy = total_reward / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS * TEST_EPISODE)
            
            print("test accuracy : ", test_accuracy)
            mean_loss = np.mean(loss_list)
            mean_embedding_loss = np.mean(embedding_loss_list)
            mean_relation_loss = np.mean(relation_loss_list)
            
            print(f'mean loss : {mean_loss}')            
            writer.add_scalar('1.embedding loss', mean_embedding_loss, episode + 1)
            writer.add_scalar('2.relation loss', mean_relation_loss, episode + 1)
            writer.add_scalar('loss', mean_loss, episode + 1)            
            writer.add_scalar('test accuracy', test_accuracy, episode + 1)
            
            loss_list = []            
            embedding_loss_list = []
            relation_loss_list = []
            
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(
                    feature_encoder.state_dict(),
                    str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                torch.save(
                    relation_network.state_dict(),
                    str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy    
    
            
if __name__ == "__main__":
    main()
