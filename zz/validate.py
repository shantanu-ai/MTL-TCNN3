import torch
import torch.utils.data
from torchvision.utils import save_image

from zz.autoEncoder import Autoencoder


class validate_Auto_encoder:
    @staticmethod
    def validate_auto_encoder(data_loader_test_list, model_path_bn, task, device):
        print(task + "----- started -----------")

        split_id = 0
        for data_loader in data_loader_test_list:
            split_id += 1
            print('-' * 50)
            print("Split: {0} =======>".format(split_id))
            model_path = model_path_bn.format(split_id)
            print("Model: {0}".format(model_path))
            model = Autoencoder().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            idx = 1
            for batch in data_loader:
                if task == "---validating":
                    img, _ = batch
                elif task == "---testing":
                    img = batch[0]

                img = img.to(device)

                outputs = model(img)
                save_image(img, './Saved_Images/' + str(idx) + '_original_test_input.jpg')
                save_image(outputs, './Saved_Images/' + str(idx) + '_test_reconstruction.jpg')
                idx += 1
            break

        print(task + "----- completed -----------")
