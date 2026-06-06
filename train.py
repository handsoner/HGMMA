import os
import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.AMNTDDA import AMNTDDA
from metric import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    # parser.add_argument('--dataset', default='metadis_meanfusion', help='dataset') #
    # parser.add_argument('--dataset', default='metadis', help='dataset')
    parser.add_argument('--dataset', default='metadis_external', help='dataset') #外部验证
    parser.add_argument('--dropout', default='0.40', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='4', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='1', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='256', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='4', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='256', type=int,
                        help='heterogeneous graph transformer output dimension')
    parser.add_argument('--hgms_lambda', default=0.1, type=float,
                        help='weight of HGMS auxiliary loss')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')
    parser.add_argument('--out_ft', type=int, default=128)
    parser.add_argument('--g_dim', type=int, default=256)
    parser.add_argument('--g_equidim', type=int, default=256)
    parser.add_argument('--p_equidim', type=int, default=256)
    parser.add_argument("--alpha", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--beta", default=0.1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--gamma", default=1,
                        help="Reconstruction error coefficient", type=float)
    parser.add_argument("--eta", default=1,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument("--lambbda", default=10,
                        help="Independence constraint coefficient", type=float)
    parser.add_argument('--loss_rate', default='0.5', type=float,
                        help='loss rate of unsupervised learning and training')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    data = get_data(args)
    args.meta_number = data['meta_number']
    args.micro_number = data['micro_number']
    args.drug_number = args.meta_number
    args.disease_number = args.micro_number
    # args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    meta_meta_graph, micro_micro_graph, data = build_similarity_graphs(data, args)
    het_mat = np.vstack((np.hstack((data['meta_sim'], data['adj'])), np.hstack((data['adj'].T, data['micro_sim']))))
    het_mat = torch.tensor(het_mat, dtype=torch.float32, device=device)

    adj_mat = construct_adj_mat(data['adj'])
    edge_idx = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
    Heter_adj_edge_index = get_edge_index_torch(adj_mat)

    meta_meta_graph = meta_meta_graph.to(device)
    micro_micro_graph = micro_micro_graph.to(device)

    train_data = {}
    Heter_adj = het_mat.float()
    train_data['Adj'] = {'data': Heter_adj, 'edge_index': Heter_adj_edge_index}
    train_data['Y_train'] = torch.DoubleTensor(data['adj'])
    train_data['feature'] = torch.FloatTensor(adj_mat)

    # drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    # disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    # protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_meta_micro']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs, accuracys, precisions, recalls, f1s, mccs, spes = [], [], [], [], [], [], [], []
    f1_score, accuracy2, recall2, precision2 = [], [], [], []
    print('Dataset:', args.dataset)

    truth = []
    probability = []
    # 存储每个fold的结果
    fold_results = []

    for i in range(args.k_fold):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print('fold:', i)
        print(Metric)

        model = AMNTDDA(args)
        if i==0:
            print(model)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc, best_spe = 0, 0, 0, 0, 0, 0, 0, 0
        best_labels, best_scores = [], []
        meta_micro_train = torch.LongTensor(data['meta_micro_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        meta_micro_test = torch.LongTensor(data['meta_micro_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        # meta_micro_protein_graph, data = build_meta_micro_protein_graph(data, data['meta_micro_train'][i], args)
        # meta_micro_protein_graph = meta_micro_protein_graph.to(device)

        meta_micro_graph, data = build_meta_micro_graph(data, data['meta_micro_train'][i], args)
        meta_micro_graph = meta_micro_graph.to(device)

        for epoch in range(args.epochs):
            model.train()
            loss_h, train_score = model(meta_meta_graph, micro_micro_graph, het_mat, edge_idx, adj_mat,
                                        meta_micro_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            loss_all = args.loss_rate * loss_h + (1 - args.loss_rate) * train_loss
            # loss_all = train_loss
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # loss_h = loss_h.detach().cpu().numpy()
            # train_loss = train_loss.detach().cpu().numpy()

            with torch.no_grad():
                model.eval()
                _, test_score = model(meta_meta_graph, micro_micro_graph, het_mat, edge_idx, adj_mat,
                                      meta_micro_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)
            
            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc, spe = get_metric(Y_test, test_score, test_prob)
            # f1_score, accuracy2, recall2, precision2 = get_metrics2(Y_test, test_score)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                    round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5), round(spe, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc, best_spe = AUPR, accuracy, precision, recall, f1, mcc, spe
                best_labels = Y_test
                best_scores = test_prob

                torch.save(model.state_dict(),"./case_study/train_model.pth")   #保存模型，按照这个逻辑只保存最后的一折的训练模型

                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        # truth.extend(best_labels)
        # probability.extend(best_scores)

        # 保存每个fold的结果
        fold_results.append((best_labels, best_scores))

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)
        accuracys.append(best_accuracy)
        precisions.append(best_precision)
        recalls.append(best_recall)
        f1s.append(best_f1)
        mccs.append(best_mcc)
        spes.append(best_spe)
        del model, optimizer, meta_micro_train, Y_train, meta_micro_test, meta_micro_graph
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # # 保存所有fold的结果
    root_path = os.path.join('results', 'GHTMDA_108')
    os.makedirs(root_path, exist_ok=True)
    # 绘制ROC曲线、AUPR曲线

    # np.save(os.path.join(root_path, 'fold_results.npy'), fold_results)
    fold_results_array = np.array(fold_results, dtype=object)
    np.save(os.path.join(root_path, 'fold_results.npy'), fold_results_array)

    # np.save('result/GHTMDA/truth.npy', np.array(truth))
    # np.save('result/GHTMDA/pred.npy', np.array(probability))

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('accuracy:', accuracys)
    accuracy_mean = np.mean(accuracys)
    accuracy_std = np.std(AUPRs)
    print('Mean accuracy:', accuracy_mean, '(', accuracy_std, ')')

    print('precision:', precisions)
    precision_mean = np.mean(precisions)
    precision_std = np.std(precisions)
    print('Mean precision:', precision_mean, '(', precision_std, ')')

    print('recall:', recalls)
    recall_mean = np.mean(recalls)
    recall_std = np.std(recalls)
    print('Mean recall:', recall_mean, '(', recall_std, ')')

    print('f1:', f1s)
    f1_mean = np.mean(f1s)
    f1_std = np.std(f1s)
    print('Mean f1:', f1_mean, '(', f1_std, ')')

    print('mcc:', mccs)
    mcc_mean = np.mean(mccs)
    mcc_std = np.std(mccs)
    print('Mean mcc:', mcc_mean, '(', mcc_std, ')')

    print('mcc:', spes)
    spe_mean = np.mean(spes)
    spe_std = np.std(spes)
    print('Mean spe:', spe_mean, '(', spe_std, ')')

    plot_roc_curves(root_path, fold_results)
    plot_pr_curves(root_path, fold_results)
    plot_combined_curves(root_path, fold_results)


    # 创建一个函数来格式化平均值和标准差
    def format_mean_std(mean, std):
        return f"{round(mean, 4):.4f} ({round(std, 4):.4f})"


    # 创建一个字典来存储所有的结果
    results = {
        'Metric': ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity'],
        'Values': [AUCs, AUPRs, accuracys, precisions, recalls, f1s, mccs, spes],
        'Mean (Std)': [
            format_mean_std(AUC_mean, AUC_std),
            format_mean_std(AUPR_mean, AUPR_std),
            format_mean_std(accuracy_mean, accuracy_std),
            format_mean_std(precision_mean, precision_std),
            format_mean_std(recall_mean, recall_std),
            format_mean_std(f1_mean, f1_std),
            format_mean_std(mcc_mean, mcc_std),
            format_mean_std(spe_mean, spe_std)
        ]
    }

    # 创建DataFrame
    df = pd.DataFrame(results)

    # 将DataFrame保存为Excel文件
    # 保存性能指标到Excel文件
    performance_metrics_file = os.path.join(root_path, 'performance_metrics.xlsx')
    df.to_excel(performance_metrics_file, index=False)

    # 如果您想保存单独的sheet来显示每次交叉验证的结果
    with pd.ExcelWriter(os.path.join(root_path, 'detailed_results.xlsx')) as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)

        for i, values in enumerate(zip(AUCs, AUPRs, accuracys, precisions, recalls, f1s, mccs, spes)):
            fold_df = pd.DataFrame({
                'Metric': ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Specificity'],
                'Value': [f"{round(v, 4):.4f}" for v in values]  # 格式化每个值为8位小数
            })
            fold_df.to_excel(writer, sheet_name=f'Fold_{i + 1}', index=False)

    print("结果已保存到Excel文件中，平均值显示为8位小数。")
