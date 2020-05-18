import numpy as np
import torch
import torch.utils.data
from torchvision.utils import save_image


class validate_Auto_encoder:
    @staticmethod
    def validate_auto_encoder(test_data_set, model, task):
        print(task + "----- started -----------")
        data_loader = torch.utils.data.DataLoader(
            test_data_set, num_workers=1, shuffle=False, pin_memory=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        noise_factor = 0.5
        idx = 1
        for batch in data_loader:
            if task == "---validating":
                img, _ = batch
            elif task == "---testing":
                img = batch[0]

            img_noisy = img + noise_factor * torch.randn(img.shape)
            img_noisy = np.clip(img_noisy, 0., 1.)
            img = img.to(device)
            img_noisy = img_noisy.to(device)

            outputs = model(img_noisy)
            save_image(img_noisy, './Saved_Images/' + str(idx) + '_noisy_test_input.png')
            save_image(outputs, './Saved_Images/' + str(idx) + '_denoised_test_reconstruction.png')
            idx += 1

        print(task + "----- completed -----------")
