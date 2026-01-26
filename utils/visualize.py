import matplotlib.pyplot as plt
def show_sample(image, mask, title=None):
    """
       Display a microscopy image and its mask side by side.

       Args:
           image: torch.Tensor or np.ndarray, shape [1,H,W] or [H,W]
           mask: torch.Tensor or np.ndarray, shape [1,H,W] or [H,W]
           title: optional figure title
       """

    # Conver torch.Tensor to numpy if needed
    if 'torch' in str(type(image)):
        image = image.squeeze().numpy()
    if 'torch' in str(type(mask)):
        mask = mask.squeeze().numpy()

    masked_image = image * mask

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].imshow(image,cmap='gray')
    ax[0].set_title("Microscopy Image")
    ax[1].imshow(masked_image,cmap='gray')
    ax[1].set_title("Microscopy Mask")

    for a in ax:
        a.axis('off')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()