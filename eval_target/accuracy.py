"""
created on:2022/6/30 16:24
@author:caijianfeng
"""
from eval_target.results import result

results = result()
label_path = '../data/prepro_flevoland4/pre_data/label.xlsx'
predict_path = '../data/prepro_flevoland4/predict_labels_CNN_7.xlsx'
acc_CNN1 = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                      predict_path=predict_path,
                                                      dim=(7, 7))

predict_path = '../data/prepro_flevoland4/predict_labels_DCNN_7.xlsx'
acc_DCNN1 = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                       predict_path=predict_path,
                                                       dim=(7, 7))
predict_path = '../data/prepro_flevoland4/predict_labels_CNN.xlsx'
acc_CNN = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                      predict_path=predict_path,
                                                      dim=(9, 9))

predict_path = '../data/prepro_flevoland4/predict_labels_DCNN.xlsx'
acc_DCNN = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                       predict_path=predict_path,
                                                       dim=(9, 9))
predict_path = '../data/prepro_flevoland4/predict_labels_GoogleNet.xlsx'
acc_GoogleNet = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                            predict_path=predict_path,
                                                            dim=(9, 9))
predict_path = '../data/prepro_flevoland4/predict_labels_ResNet.xlsx'
acc_ResNet = results.accuracy_readme_json_without_label0(label_path=label_path,
                                                         predict_path=predict_path,
                                                         dim=(9, 9))
print('acc_CNN1:', acc_CNN1,
      '\nacc_DCNN1:', acc_DCNN1,
      '\nacc_CNN:', acc_CNN,
      '\nacc_DCNN:', acc_DCNN,
      '\nacc_GoogleNet:', acc_GoogleNet,
      '\nacc_ResNet:', acc_ResNet)


