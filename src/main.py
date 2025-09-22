import os
import argparse
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src import dataio, sampling, siren, utils

def train_model(coords, values, dims, sampling_mask, outdir, method, sampling_pct,
                num_hidden=11, neurons=120, max_epochs=500, lr=1e-5,
                decay_interval=5, decay_rate=0.8, patience=10):
    """
    Train Residual SIREN on sampled data and reconstruct.
    """
    os.makedirs(outdir, exist_ok=True)

    # Apply sampling mask
    sampled_coords = coords[sampling_mask > 0]
    sampled_vals = values[sampling_mask > 0]

    # Normalize
    sampled_vals, lo_val, hi_val = utils.normalize_m1_p1(sampled_vals)
    coords_norm = coords.copy()
    coords_norm[:, 0] = 2.0 * (coords[:, 0] / (dims[0] - 1) - 0.5)
    coords_norm[:, 1] = 2.0 * (coords[:, 1] / (dims[1] - 1) - 0.5)
    coords_norm[:, 2] = 2.0 * (coords[:, 2] / (dims[2] - 1) - 0.5)

    # Torch data
    torch_coords = torch.from_numpy(coords_norm).float()
    torch_vals = torch.from_numpy(values).float()
    torch_sampled = TensorDataset(torch.from_numpy(sampled_coords).float(),
                                  torch.from_numpy(sampled_vals).float())
    loader = DataLoader(torch_sampled, batch_size=2048, shuffle=True, pin_memory=True)

    # Model
    model = siren.ResidualSirenNet(num_hidden_layers=num_hidden, neurons_per_layer=neurons)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    min_loss = float("inf")
    best_wts = None
    patience_counter = 0
    lr_current = lr

    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in loader:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss={avg_loss:.6f}")

        # LR decay
        if (epoch + 1) % decay_interval == 0:
            lr_current *= decay_rate
            for g in optimizer.param_groups:
                g['lr'] = lr_current

        # Early stopping
        if avg_loss < min_loss:
            min_loss = avg_loss
            best_wts = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Save best model
    model.load_state_dict(best_wts)
    model.eval()
    model_path = os.path.join(outdir, f"residual_sampling_{method}_{sampling_pct}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Reconstruction
    recon = []
    group = 1024
    with torch.no_grad():
        for i in range(0, torch_coords.shape[0], group):
            xb = torch_coords[i:i+group].to(device)
            preds = model(xb).cpu().numpy().flatten()
            recon.append(preds)
    recon = np.concatenate(recon)

    # Denormalize
    recon = utils.denormalize_m1_p1(recon, lo_val, hi_val)

    return recon

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="synthetic",
                        help="synthetic | vti")
    parser.add_argument("--vti_path", type=str, default=None,
                        help="Path to .vti file (if dataset=vti)")
    parser.add_argument("--method", type=str, default="random",
                        help="random | hist_based | hist_grad | random_hist_grad | random_hist_based | adaptive | void_cluster")
    parser.add_argument("--sampling", type=float, default=5.0,
                        help="Sampling percentage")
    parser.add_argument("--outdir", type=str, default="outputs/",
                        help="Output directory")
    args = parser.parse_args()

    if args.dataset == "synthetic":
        vals = utils.synthetic_scalar_field(64, 64, 64)
        dims = vals.shape
        coords = np.array([[i, j, k] for k in range(dims[2])
                                      for j in range(dims[1])
                                      for i in range(dims[0])])
        values = vals.ravel()
    elif args.dataset == "vti":
        if args.vti_path is None:
            raise ValueError("Must provide --vti_path for dataset=vti")
        data, dims, name, vals_np, x, y, z = dataio.read_vti(args.vti_path)
        coords = np.stack([x, y, z], axis=1)
        values = vals_np
    else:
        raise ValueError("Unknown dataset")

    # Sampling
    ratio = args.sampling / 100.0
    if args.method == "random":
        mask = sampling.random_sampling(ratio, values.size)
    elif args.method == "hist_based":
        mask = sampling.hist_based_sampling(ratio, values)
    elif args.method == "hist_grad":
        mask = sampling.hist_grad_sampling(ratio, values, dims)
    elif args.method == "random_hist_grad":
        mask = sampling.random_hist_grad_sampling(ratio, values, dims)
    elif args.method == "random_hist_based":
        mask = sampling.random_hist_based_sampling(ratio, values)
    elif args.method == "adaptive":
        mask = sampling.adaptive_sampling(values, target_pct=args.sampling)
    elif args.method == "void_cluster":
        total_points = int(values.size * ratio)
        mask = sampling.void_cluster_simplified(values, total_points)
    else:
        raise ValueError("Unknown method")

    # Train + reconstruct
    recon = train_model(coords, values, dims, mask,
                        args.outdir, args.method, args.sampling)

    # Evaluate
    psnr_val = utils.psnr(values, recon)
    print(f"PSNR: {psnr_val:.4f} dB")

    # Save reconstruction (only if VTI dataset)
    if args.dataset == "vti":
        out_vti = os.path.join(args.outdir, f"recon_{args.method}_{args.sampling}.vti")
        dataio.write_reconstruction_vti(data, recon, out_vti)
        print(f"Reconstruction saved: {out_vti}")

if __name__ == "__main__":
    main()
