import torch
import numpy as np
import task_generator as tg
import omniglot_train_few_shot as ot
import os
import argparse
import random
import scipy as sp
import scipy.stats


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=10)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
HIDDEN_UNIT = args.hidden_unit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # * Step 1: init data folders
    print("init data folders")
    
    # * init character folders for dataset construction
    metartrain_character_folders, metatest_character_folders = tg.omniglot_character_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = ot.CNNEncoder().to(device)
    relation_network = ot.RelationNetwork(FEATURE_DIM, RELATION_DIM).to(device)
    
    feature_encoder.eval()
    relation_network.eval()    
    
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
        
    if os.path.exists(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")
        
    total_accuracy = 0.0
    for episode in range(EPISODE):
        # * test
        print("Testing...")
        total_rewards = 0
        accuracies = []
        
        for i in range(TEST_EPISODE):
            degrees = random.choice([0, 90, 180, 270])
            task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS)
            
            sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
            test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test", shuffle=True, rotation=degrees)
            
            sample_images, sample_labels = next(iter(sample_dataloader))
            test_images, test_labels = next(iter(test_dataloader))

            sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            # * Calculate features
            sample_features = feature_encoder(sample_images)            
            sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            test_features = feature_encoder(test_images)
            
            sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
            test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)
            
            relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
            relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
            
            _, predict_labels = torch.max(relations.data, 1)
            
            rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
            
            total_rewards += np.sum(rewards)
            accuracy = np.sum(rewards) / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS)
            accuracies.append(accuracy)
            
        test_accuracy, h = mean_confidence_interval(accuracies)
        
        print(f'test accuracy : {test_accuracy}, h : {h}')
        total_accuracy += test_accuracy
        
    print(f"average accuracy : {total_accuracy / EPISODE}")
    

if __name__ == "__main__":
    main()
              
    
    
    