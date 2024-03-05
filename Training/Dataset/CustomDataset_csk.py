import rasterio
import torch, torchvision
import os
import json
from torch.utils.data import Dataset

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MS3Dataset_CSK(Dataset):
    def __init__(self, data_path, coco_path, real_conv:bool, transform: bool = True, dtype = torch.float16):
        super().__init__()

        self.fold_path = data_path
        self.coco_data = self.loader_coco(coco_path)
        self.transform = transform
        self.dtype = dtype
        self.real_conv = real_conv

    def __len__(self):
        return len(self.coco_data["images"])

    def __getitem__(self, index):

        file_name, img_id = self.fname_and_id_from_index(self.coco_data, index)
        path_img = self.build_path_img(file_name)
        x_real_vh, x_imag_vh = self.loader_img(file_path=path_img)

        target = []
        inc_angle = []

        for annotation in self.coco_data["annotations"]:

            if annotation["image_id"] == img_id:

                ann_to_add = annotation["centroid"]

                target.append(ann_to_add)

                if not inc_angle:
                    inc_angle.append(annotation["incident_angle"])

        x_real_vh = torch.from_numpy(x_real_vh)
        x_imag_vh = torch.from_numpy(x_imag_vh)
                            
        target = torch.tensor(target, dtype=self.dtype)
        inc_angle = torch.tensor(inc_angle, dtype=self.dtype)

        if not self.real_conv:
            sample = torch.stack((x_real_vh, x_imag_vh))       
            if self.transform:
                sample = torchvision.transforms.Normalize([-0.0009, 0.0003], [0.1484, 0.1482])(sample)
        else:
            sample = torch.sqrt(x_real_vh **2 + x_imag_vh **2).unsqueeze(0)
            if self.transform:
                sample = torchvision.transforms.Normalize(0.1222, 0.1718)(sample)

        return {"sample": sample.to(dtype=self.dtype), "target": target, "inc_angle": inc_angle}

    def loader_img(self, file_path):

        try:

            with rasterio.open(file_path) as f:
                x_real_vh = f.read(1)
                x_imag_vh = f.read(2)

            return x_real_vh, x_imag_vh

        except Exception as e:
            print(f"Error occured: Could not open {file_path}: {e}")
            raise e

    def build_path_img(self, filename):

        img_idx = filename.split("Im_")[1].split("_PROD")[0]
        prod_idx = filename.split("PROD_")[1].split("_PIN")[0]
        pin_idx = filename.split("PIN_")[1].split("_SUB")[0]
        sub_idx = filename.split("Cal_")[1].split(".dim")[0]

        pin_folder = os.path.join(
            self.fold_path, img_idx, str(int(img_idx)) + "-" + prod_idx, "outdir_py" + prod_idx, "Pin" + pin_idx)

        for file in os.listdir(pin_folder):

            if sub_idx in file and ".tif" in file:

                path_file = file

        return os.path.join(pin_folder, path_file)

    @staticmethod
    def fname_and_id_from_index(data, index):

        fname = data["images"][index]["file_name"]
        img_id = data["images"][index]["id"]

        return fname, img_id

    @staticmethod
    def loader_coco(path_coco):

        with open(path_coco, 'r') as file_coco:

            data = json.load(file_coco)

        return data
