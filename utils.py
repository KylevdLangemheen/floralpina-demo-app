from torchvision import transforms
from torchvision import models
from PIL import Image
import torch
import numpy as np
from typing import List

class CenterCropMin:
    """Crop the center square using the smallest image dimension.
    
    Simply finds the smallest dimension along heigh and width and
    crops a square of that dimension around the center. This way,
    non-square images can be accepted so long as the object of
    interest is (mostly) centered.
    """
    def __call__(self, img: Image.Image) -> Image.Image:
        """Forward call for the transform.
        
        Accepts an image returns the largest cropped square around
        the center.
        
        Args:
            img:
                A PIL Image.
        
        Returns:
            The cropped PIL image."""
        w, h = img.size
        size = min(w, h)
        return transforms.functional.center_crop(img, size)

def load_model(path: str) -> torch.nn.Module:
    """Loads model and apply state dict from path and returns loaded Module.
    
    Args:
        path:
            Path to the final layer state dict.
        
    Returns:
        Model with finetuned weights set to eval."""
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.classifier[6]=torch.nn.Linear(model.classifier[6].in_features, 102) # <- we need to replace the last layer
    model.classifier[6].load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def predict_one(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Get predicted probabilities from the model on a single input.
    
    Args:
        model:
            The classifier, a finetuned torch model.
        x:
            A single (cropped, resized) input image.
            
    Returns:
        A tensor containing the softmax probabilities across 102 flower classes."""
    x = input_transform(x)
    y_ = model(x.unsqueeze(0)).squeeze(0)
    p = torch.nn.functional.softmax(y_, dim=0).detach().cpu()
    return p

def get_prediction(p: torch.Tensor) -> str:
    """Helper function to get the most probable class in str format.
    
    Args:
        p:
            Input softmax probabilities.
    
    Returns:
        String of the most probable class name."""
    return classes[p.argmax()]

def iter_top_proba_predictions(p: torch.Tensor, n_sigma: int | float=2):
    """Yield indices of top probabilities, limited by n_sigma.
    
    This relies on the assumption that every flower is equally likely.
    We then get a 1/102 chance of being correct with a straightforward
    spread. From this we can calculate how confident we are that our
    prediction is n_sigma standard deviations away from guessing.
    
    Args:
        p:
            Input softmax probabilities.
        n_sigma:
            How many std away from guessing we want to be.

    Yields:
        Indices of top probabilities starting at the most confident prediction and
        stopping at the least confident prediction that is still within threshold
    """
    mu = 1/len(classes) # Guessing
    sigma = np.sqrt(mu * (1-mu))
    threshold = mu + n_sigma * sigma
    for i in torch.argsort(p, descending=True):
        if p[i] < threshold:
            break

        yield i

def iter_top_n_predictions(p: torch.Tensor, n: int=5):
    """Yield indices of top n predictions.
    
    Args:
        p:
            Input softmax probabilities.
        n:
            Max indices to return.
    
    Yields:
        Indices of top probabilities starting at the most confident prediction and
        stopping after yielding n indices or reaching the end."""
    for i in torch.argsort(p, descending=True)[:n]:
        yield i

# Input transform to further transform inputs to the model
input_transform = models.VGG16_Weights.DEFAULT.transforms()

# TODO: store classes in config file, implement test to assert equivalence with dataset.
classes = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia?",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]