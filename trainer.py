import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import CarvanaDataset
import matplotlib.pyplot as plt

def plot_gradient_flow(model):
    ave_grads = []
    layers = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())

    plt.figure(figsize=(10, 5))
    plt.bar(layers, ave_grads, color="b", alpha=0.6)
    plt.hlines(0, 0, len(ave_grads), color="k",
               linestyle="dashed", linewidth=0.5)
    plt.xticks(rotation=90)
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient Magnitude")
    plt.title("Gradient Flow")
    plt.savefig("figures/graph.png")
    plt.close()

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True,):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds, _ = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}'
    )
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()


def save_predictions_as_imgs(
        loader, model, folder, device='cuda'
):
    saved_images_folder = folder + 'saved_images'
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds, expert_outputs = model(x)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f'{saved_images_folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f'{saved_images_folder}/truth_{idx}.png'
        )
        for layer in range(len(expert_outputs)):
            for expert in range(len(expert_outputs[0])):
                torchvision.utils.save_image(
                    expert_outputs[layer][expert], f'{folder}/expert_outputs/{idx}_{layer}_{expert}.png')


def train_fn(loader, model, optimizer, loss_fn1, loss_fn2, scaler):

    loop = tqdm(loader)
    losses = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to('cuda')

        targets = targets.float().unsqueeze(1).to('cuda')

        # forward

        with torch.amp.autocast('cuda'):

            predictions, expert_output = model(data)
            loss1, loss2 = loss_fn1(predictions, targets), loss_fn2(expert_output)
            losses.append((loss1, loss2))
            loss = loss1 + loss2


        # backward

        optimizer.zero_grad()

        scaler.scale(loss).backward()
        
        # plot_gradient_flow(model)

        scaler.step(optimizer)

        scaler.update()

        # update tqdm loop

        loop.set_postfix(loss=loss.item())
            
    return losses
