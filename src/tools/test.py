import numpy as np
import os

def prepare_dataset_test(batch_size, data_path='Path/to/GraphiContact/datasets/New_Damon/test',
                        label_path='Path/to/GraphiContact/datasets/New_Damon/rhobin_damon_test_release.npz'):
    
    org_img_paths = os.listdir(data_path)
    org_img_paths = [os.path.join(data_path, path) for path in org_img_paths]
    
    test_label = np.load(label_path)
    test_imgs_names = test_label['imgname']
    test_imgs_names = [img.split('/')[-1] for img in test_imgs_names]
    test_3d_con_label = test_label['contact_label']

    print("Total test images:", len(org_img_paths))
    print("Total test labels:", len(test_3d_con_label))

    test_data = []
    contact_label = []
    for img_path in org_img_paths:
        for i in range(len(test_imgs_names)):
            if test_imgs_names[i] in img_path:
                test_data.append(img_path)
                contact_label.append(test_3d_con_label[i])
    test_data = [data for data in test_data if len(test_data) >= 5]
    contact_label = [data for data in contact_label if len(data) == 6890]

    print('Filtered test length: ' + str(len(test_data)))
    print('Filtered contact length: ' + str(len(contact_label)))

    batch_input = []
    for i in range(0, int(len(test_data)/batch_size)):
        data_batch = test_data[i*batch_size:(i+1)*batch_size]
        label_batch = contact_label[i*batch_size:(i+1)*batch_size]
        print(f"Batch {i}: Data batch length: {len(data_batch)}, Label batch length: {len(label_batch)}")
        batch_input.append((data_batch, label_batch))
    
    return iter(batch_input)

# Test the prepare_dataset_test function
batch_size = 8
data_path = 'Path/to/GraphiContact/datasets/New_Damon/test'
label_path = 'Path/to/GraphiContact/datasets/New_Damon/rhobin_damon_test_release.npz'

test_data_iter = prepare_dataset_test(batch_size, data_path, label_path)

try:
    test_data_batch = next(test_data_iter)
    print("First batch of test data loaded successfully.")
    print("Data batch length:", len(test_data_batch[0]))
    print("Label batch length:", len(test_data_batch[1]))
except StopIteration:
    print("No data available in test dataset.")
