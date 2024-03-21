from pathlib import Path
import mlx.data as dx

def read(path: str) -> str:
    with open(path, 'r') as file:
        return ''.join(file.readlines())

def normalize(x):
    x = x.astype("float32") / 255.0
    return (x - 0.5) / 0.5

class Dataset:
    def __init__(
        self,
        instance_data_root: str,
        instance_prompt: str, # e.g. "photo of a ewpp person"
        tokenizer,
        size=512
    ):
        self.size = size
        self.tokenizer = tokenizer

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Root path for instance data does not exist")

        self.instance_images_path = list(self.instance_data_root.glob("*.jpg"))
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        samples = [
            dict(
                file=str(img.relative_to(self.instance_data_root)).encode(),
                label=self.instance_prompt.encode()
            )
            for img in sorted(self.instance_images_path)
        ]

        print(samples)
        print(self.instance_data_root)

        dset = dx.buffer_from_vector(samples).load_image(
            "file", prefix=str(self.instance_data_root), output_key="image"
        )

        self.image_transforms = (
            dset.image_resize("image", size, size)
            .image_center_crop("image", size, size)
            .key_transform("image", normalize)
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {
            "instance_images": self.image_transforms[index % self.num_instance_images]["image"],
            "instance_prompt_ids": self.tokenizer.tokenize(self.instance_prompt)
        }

        return example
