from data_process.data_preprocess import Data

data_prepro = Data()

data_prepro.save_data_label_segmentation(data_path='./work.xlsx',
                                         label_path='./label.xlsx',
                                         target_path='./data/data_TR',
                                         train_list_path='./train_list.txt',
                                         eval_list_path='./eval_list.txt',
                                         patch_size=(3, 3))
