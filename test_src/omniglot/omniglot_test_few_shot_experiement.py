import torch
import numpy as np
import task_generator_no_args as tg
import omniglot_train_few_shot_experiement as ot
import os
import random
import scipy as sp
import scipy.stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

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
    RFT = RandomForestClassifier(n_estimators=100, random_state=1, warm_start=True)
    relation_network = ot.RelationNetwork(FEATURE_DIM, RELATION_DIM).to(device)
    
    feature_encoder.eval()
    relation_network.eval()    
    
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
    
    if os.path.exists(str("./models/omniglot_random_forest_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        RFT = pickle.load(open(str("./models/omniglot_random_forest_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl"), 'rb'))
        print("load random forest success")
        
    if os.path.exists(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")

    RFT_total_accuracy = 0.0
    total_accuracy = 0.0
    soft_voting_total_accuracy = 0.0
    
    for episode in range(1):
        # * test
        print("Testing...")
        total_rewards = 0
        soft_voting_total_rewards = 0
        
        accuracies = []
        RFT_accuracies = []
        soft_voting_accuracies = []
        
        for i in range(100):
            degrees = random.choice([0, 90, 180, 270])
            task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, SAMPLE_NUM_PER_CLASS)
            
            sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
            test_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="test", shuffle=True, rotation=degrees)
            
            sample_images, sample_labels = next(iter(sample_dataloader))
            test_images, test_labels = next(iter(test_dataloader))

            RFT_samples, RFT_sample_labels = sample_images, sample_labels
            RFT_test, RFT_test_labels = test_images, test_labels
        
            sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
            test_images, test_labels = test_images.to(device), test_labels.to(device)
            
            # * Calculate features
            _, sample_features = feature_encoder(sample_images)            
            sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            _, test_features = feature_encoder(test_images)
            
            sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
            test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            test_features_ext = torch.transpose(test_features_ext, 0, 1)
            
            relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
            relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
            
            RFT_predict = RFT.predict(relations.detach().cpu())
            assert RFT_predict.shape == test_labels.detach().cpu().shape
            RFT_score = accuracy_score(RFT_predict, test_labels.detach().cpu())

            
            _, predict_labels = torch.max(relations.data, 1)
            rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
            total_rewards += np.sum(rewards)
            accuracy = np.sum(rewards) / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS)
            
            RFT_prob = RFT.predict_proba(relations.detach().cpu())
            relation_prob = torch.softmax(relations.data, dim=1)
          
            RFT_prob_tensor = torch.tensor(RFT_prob).to(device)
            # soft_voting = (RFT_prob_tensor + relation_prob) / 2
            soft_voting = (RFT_prob_tensor / relation_prob)
            _, soft_voting_predicted_labels = torch.max(soft_voting, 1)
            
            '''
            for i in range(len(RFT_predict)):
                if RFT_predict[i] != predict_labels[i]:
                    print(f'RFT_prob      : {RFT_prob[i]}, label : {RFT_predict[i]}, answer : {test_labels[i]}')
                    print(f'relation_prob : {relation_prob[i].detach().cpu().numpy()}, label : {predict_labels[i]}, answer : {test_labels[i]}')
                    print(f'soft_voting   : {soft_voting[i].detach().cpu().numpy()}, label : {soft_voting_predicted_labels[i]}, answer : {test_labels[i]}')
                    print("----------------------------------")
            '''
            '''
                print(relations.data)
                print(RFT_prob)
                print(relation_prob)
                
                assert 1 == 2
            '''
            
            soft_voting_rewards = [1 if soft_voting_predicted_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
            soft_voting_total_rewards += np.sum(soft_voting_rewards)
            soft_voting_accuracy = np.sum(soft_voting_rewards) / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS)
            
            print(f"{i+1}th RFT accuracy : {RFT_score}, CNN accuracy : {accuracy}, combined_accuracy : {soft_voting_accuracy}")
            
            RFT_accuracies.append(RFT_score)
            accuracies.append(accuracy)
            soft_voting_accuracies.append(soft_voting_accuracy)
            
        RFT_test_accuracy, RFT_h = mean_confidence_interval(RFT_accuracies)
        test_accuracy, h = mean_confidence_interval(accuracies)
        soft_voting_test_accuracy, soft_voting_h = mean_confidence_interval(soft_voting_accuracies)
        
        print(f'RFT_test_accuracy accuracy : {RFT_test_accuracy:.4f}, h : {RFT_h:.4f}')
        print(f'test accuracy : {test_accuracy:.4f}, h : {h:.4f}')
        print(f'test soft_voting_test_accuracy : {soft_voting_test_accuracy:.4f}, h : {soft_voting_h:.4f}')
        
        RFT_total_accuracy += RFT_test_accuracy
        total_accuracy += test_accuracy
        soft_voting_total_accuracy += soft_voting_test_accuracy
        
    print(f"average RFT_total_accuracy : {RFT_total_accuracy :.4f}")
    print(f"average accuracy : {total_accuracy :.4f}")
    print(f"soft_voting_total_accuracy : {soft_voting_total_accuracy :.4f}")
    

if __name__ == "__main__":
    main()
              
    
    
    